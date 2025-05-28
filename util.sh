#!/bin/bash

declare -A jobnames
for job in $(qstat -u iferreira | awk 'NR>5 {print $1}'); do
    name=$(qstat -f "$job" | awk -F'= ' '/Job_Name =/ {print $2}' | tr -d ' ')
    if [[ -n "$name" ]]; then
        echo "Currently queued: experiments/$name"
        jobnames["experiments/$name"]=1
    fi
done

deleted_any=true

while $deleted_any; do
    deleted_any=false
    while read -r dir; do
        if [[ -z "$(find "$dir" -mindepth 1 -type d)" && \
              ! -f "$dir/Accuracy.png" && \
              -z "${jobnames[$dir]}" ]]; then
            echo "Deleting $dir"
            rm -rf "$dir"
            deleted_any=true
        fi
    done < <(find experiments/* -type d)
done