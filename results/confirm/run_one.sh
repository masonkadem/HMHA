#!/bin/bash
# one confirmation job: python experiments/run.py <args> --skip-redundancy --out results/confirm
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/run.py $args --skip-redundancy --out results/confirm > "results/confirm/logs/$name.log" 2>&1
echo "done ($?): $args"
