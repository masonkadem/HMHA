.PHONY: test run sweep figures paper clean
PY ?= python
WORKERS ?= 4

test:                       ## ground truth first; nothing downstream counts without it
	pytest -q tests/

sweep: test                 ## resumable; override ONLY= and WORKERS=
	$(PY) experiments/sweep.py --run --workers $(WORKERS) $(if $(ONLY),--only $(ONLY),)

controls:                   ## the 18 runs that settle the paper's open question
	$(PY) experiments/sweep.py --run --workers $(WORKERS) --only stall_eps,slowgain_ctl

figures:                    ## regenerate every figure, table and prose macro
	$(PY) experiments/aggregate.py --results results --out figures --readme README.md
	$(PY) experiments/paper_assets.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode chapter.tex >/dev/null && \
	           pdflatex -interaction=nonstopmode chapter.tex >/dev/null
	@echo "wrote paper/chapter.pdf"

status:                     ## what has run and what has not
	@$(PY) experiments/sweep.py --list | tail -1
