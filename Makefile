# Uso:
# make pos tagger=spacy|stanza
# make train cfg=experiments/configs/baseline.yaml
# make eval cfg=experiments/configs/baseline.yaml
# make plots

PY=python
TAGGER?=spacy
CFG?=experiments/configs/baseline.yaml

pos:
	$(PY) src/features/run_pos.py --tagger $(TAGGER)

train:
	$(PY) src/main_train.py --config $(CFG)

eval:
	$(PY) src/main_eval.py --config $(CFG)

plots:
	$(PY) src/utils/make_plots.py --results_dir experiments/results
