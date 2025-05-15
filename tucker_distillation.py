from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar100
from toolbox.utils import plot_the_things, evaluate_model

import torch
import torch.optim as optim
import torch.nn.functional as F

import tensorly as tl

from pathlib import Path
import argparse

DEVICE = "cuda"
EPOCHS = 150
BATCH_SIZE = 128

tl.set_backend("pytorch")
def tucker(feature_map, ranks=[BATCH_SIZE, 32, 8, 8]): 
    core, factors = tl.decomposition.tucker(feature_map, rank=ranks)
    return core, factors

def compute_core(feature_map, factors):
    return tl.tenalg.multi_mode_dot(feature_map, [f.T for f in factors], modes=[0, 1, 2, 3])

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))


def tucker_distillation(teacher_outputs, student_outputs, targets, ranks=None):
    teacher_fmap = teacher_outputs[2]
    student_fmap = student_outputs[2]

    teacher_core , teacher_factors = tucker(teacher_fmap, ranks)
    student_core = compute_core(student_fmap, teacher_factors)

    tucker_loss = BETA * F.l1_loss(FT(teacher_core), FT(student_core))
    hard_loss = F.cross_entropy(student_outputs[3], targets)
    return tucker_loss, hard_loss

def tucker_recomp_distillation(teacher_outputs, student_outputs, targets,recomp_target,ranks=None): # Decomposes and recomposes teacher feature map
    teacher_fmap = teacher_outputs[2]
    student_fmap = student_outputs[2]

    teacher_core , teacher_factors = tucker(teacher_fmap, ranks)
    if recomp_target == 'teacher':
        teacher_reconstructed = tl.tucker_to_tensor((teacher_core, teacher_factors))
        soft_loss = BETA * F.l1_loss(FT(teacher_reconstructed), FT(student_fmap))
    elif recomp_target == 'student':
        student_core = compute_core(student_fmap, teacher_factors)
        student_reconstructed = tl.tucker_to_tensor((student_core, teacher_factors))
        soft_loss = BETA * F.l1_loss(FT(teacher_fmap), FT(student_reconstructed))
    elif recomp_target == 'both':
        student_core = compute_core(student_fmap, teacher_factors)
        teacher_reconstructed = tl.tucker_to_tensor((teacher_core, teacher_factors))
        student_reconstructed = tl.tucker_to_tensor((student_core, teacher_factors))
        soft_loss = BETA * F.l1_loss(FT(teacher_reconstructed), FT(student_reconstructed))

    hard_loss = F.cross_entropy(student_outputs[3], targets)
    return soft_loss, hard_loss

def feature_map_distillation(teacher_outputs, student_outputs, targets):
    teacher_fmap = teacher_outputs[2]
    student_fmap = student_outputs[2]
    soft_loss = BETA * F.l1_loss(FT(teacher_fmap), FT(student_fmap))
    hard_loss = F.cross_entropy(student_outputs[3], targets)
    return soft_loss, hard_loss

DISTILLATIONS = {
    'tucker' : tucker_distillation,
    'tucker_recomp': tucker_recomp_distillation,
    'featuremap': feature_map_distillation
}

parser = argparse.ArgumentParser(description='Run a training script with custom parameters.')
parser.add_argument('--beta', type=int, default='125')
parser.add_argument('--experiment_name', type=str, default='test')
args = parser.parse_args()

Distillation = DISTILLATIONS['featuremap']
Ranks = [BATCH_SIZE,32,8,8]
Recomp_target = "both"
experiment_path = "test"
BETA = args.beta

Path(f"experiments/{experiment_path}").mkdir(parents=True, exist_ok=True)
print(vars(args))

# Model setup
model_path = r"toolbox/Cifar100_ResNet112.pth"
teacher = ResNet112(100).to(DEVICE)
teacher.load_state_dict(torch.load(model_path, weights_only=True)["weights"])

student = ResNet56(100).to(DEVICE)

Data = Cifar100(BATCH_SIZE)
trainloader, testloader = Data.trainloader, Data.testloader

optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_hard_loss = []
train_soft_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0

for i in range(EPOCHS):
    print(i)
    teacher.eval()
    student.train()
    total_hard_loss, total_soft_loss, correct, total = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % 10 == 0:
            print(batch_idx)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)
        if Distillation.__name__ == 'feature_map_distillation':
            soft_loss, hard_loss = Distillation(teacher_outputs, student_outputs, targets)
        else:
            if Distillation.__name__ == 'tucker_recomp_distillation':
                soft_loss, hard_loss = Distillation(teacher_outputs, student_outputs, targets, Recomp_target, Ranks)
            else:
                soft_loss, hard_loss = Distillation(teacher_outputs, student_outputs, targets, Ranks)

        loss = soft_loss + hard_loss
        loss.backward()
        optimizer.step()

        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    scheduler.step()

    trhl, trsl = total_hard_loss/(b_idx+1), total_soft_loss/(b_idx+1)
    tra = 100*correct/total

    print(f'TRAIN | Hard Loss: {trhl:.3f} | Soft Loss {trsl:.3f} | Acc: {tra:.3f} |')
    tel, tea = evaluate_model(student, testloader)

    train_hard_loss.append(trhl)
    train_soft_loss.append(trsl)
    train_acc.append(tra)
    test_loss.append(tel)
    test_acc.append(tea)

    if tea > max_acc:
        max_acc = tea
        torch.save({'weights': student.state_dict()}, f'experiments/{experiment_path}/ResNet56.pth')
    
    plot_the_things((train_hard_loss, train_soft_loss), test_loss, train_acc, test_acc, experiment_path)

import json

with open(f'experiments/{experiment_path}/metrics.json', 'w') as f:
    json.dump({
        'train_hard_loss': train_hard_loss,
        'train_soft_loss': train_soft_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, f)