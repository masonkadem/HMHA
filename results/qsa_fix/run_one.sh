#!/bin/bash
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/run.py $args --skip-redundancy --out results/qsa_fix > "results/qsa_fix/logs/$name.log" 2>&1
echo "done ($?): $args"
