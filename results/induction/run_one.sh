#!/bin/bash
# one induction job: python experiments/induction_run.py <args>  (writes results/induction/*.pkl)
args="$*"
name=$(echo "$args" | tr ' ' '_' | tr -d '-')
.venv/Scripts/python.exe experiments/induction_run.py $args > "results/induction/logs/$name.log" 2>&1
echo "done ($?): $args"
