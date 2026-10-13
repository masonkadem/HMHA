#!/bin/bash
# one review-control job: python experiments/run.py <args> --skip-redundancy --out results/review
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/run.py $args --skip-redundancy --out results/review > "results/review/logs/$name.log" 2>&1
echo "done ($?): $args"
