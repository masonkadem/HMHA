#!/bin/bash
# one proposal job: python experiments/run.py <args> --skip-redundancy --out results/trial2
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/run.py $args --skip-redundancy --out results/trial2 > "results/trial2/logs/$name.log" 2>&1
echo "done ($?): $args"
