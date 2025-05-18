#!/bin/bash

declare -A jobnames
for job in $(qstat -u iferreira | awk 'NR>5 {print $1}'); do
    name=$(qstat -f "$job" 2>/dev/null | awk -F'= ' '/Job_Name =/ {print $2}' | tr -d ' ')
    if [[ -n "$name" ]]; then
        echo "Currently queued: experiments/$name"
        jobnames["experiments/$name"]=1
    fi
done

echo

find experiments -type d | while read -r dir; do
    if [[ -z "$(find "$dir" -mindepth 1 -type d 2>/dev/null)" && \
          ! -f "$dir/accuracy.png" && \
          -z "${jobnames[$dir]}" ]]; then
        echo "Deleting $dir"
        rm -rf "$dir"
    fi
done