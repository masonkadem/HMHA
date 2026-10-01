#!/bin/bash
# one robustness job: python experiments/run.py <args> --skip-redundancy --out results/robust
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/run.py $args --skip-redundancy --out results/robust > "results/robust/logs/$name.log" 2>&1
echo "done ($?): $args"
