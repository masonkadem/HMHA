#!/bin/bash
# one job of the equal-size comparison (every method on k* = 2, 4, 8 with 10 seeds)
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/run.py $args --skip-redundancy --out results/equal > "results/equal/logs/$name.log" 2>&1
echo "done ($?): $args"
