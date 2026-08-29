# Reproduce every number in the paper.  `make` runs the gate; `make all` regenerates
# the results files first, then gates them.  No GPU is used or needed.
PY := python3
MODEL ?= meta-llama/Llama-3.1-8B-Instruct
BATCH ?= 32
EXP := experiments

.PHONY: help verify all attack packing realdata security nu menger legacy paper clean
help:
	@echo "make verify   - gate 1: results support the claims; gate 2 (needs the paper) that it quotes them"
	@echo "make all      - regenerate all results, then verify   (~3 min)"
	@echo "make attack   - E19-E20  attack calculus + 5 adversaries   -> results_attack.json"
	@echo "make packing  - E12-E18  packing law, separation, off-graph -> results_packing.json"
	@echo "make realdata - RD1-RD2  TPC-H workload + PEGASE topology   -> results_realdata.json"
	@echo "make security - E28-E31  amplification, multi-target, adaptive, tightness"
	@echo "make nu       - nu vs tau*_A, the packing lower bound's tightness"
	@echo "make menger   - E34/E35 fractional budget and the max-flow certificate"
	@echo "make legacy   - E1-E11   supporting experiments             -> results.json"
	@echo "make llm      - E22 verdict-error correlation (NEEDS 1 GPU): MODEL=..."
	@echo "make paper    - build paper/main.pdf (needs a LaTeX install)"

verify:
	@cd $(EXP) && $(PY) verify_claims.py && $(PY) audit_paper.py

attack:   ; cd $(EXP) && $(PY) attack.py
packing:  ; cd $(EXP) && $(PY) packing.py
realdata: ; cd $(EXP) && $(PY) realdata.py
security: ; cd $(EXP) && $(PY) multitarget.py
nu:       ; cd $(EXP) && $(PY) nu_check.py
menger:   ; cd $(EXP) && $(PY) menger_check.py
legacy:   ; cd $(EXP) && $(PY) run_all.py
llm:      ; cd $(EXP) && $(PY) run_on_gpu.py --model $(MODEL) --batch $(BATCH)

all: attack packing realdata security nu menger verify

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null
	@echo "paper/main.pdf"

clean:
	rm -f paper/main.aux paper/main.log paper/main.out paper/main.toc paper/main.pdf
