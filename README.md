# TA – Pipeline de Extracción de Datos por Voz (ES)
Prototipo reproducible para la Parte 4–10 del informe (Experimentación y Resultados).

## Estructura
```
/data/               (scripts de descarga + README fuentes)
 /synthetic/         (generador y semillas; NO subir PII real)
/models/             (configs y pesos o enlaces)
/src/
  features/          (embeddings, POS, rasgos)
  stt/               (whisper.py, google_stt.py)
  ner/               (svm.py, mlp.py, eval.py)
  utils/             (split, metrics, noise_injection)
  main_train.py
  main_eval.py
/experiments/
  configs/           (yaml: grids, ablations, seeds)
  results/           (csv con métricas; figs .png)
/docs/               (tablas .md, figuras, informe .pdf)
Makefile
requirements.txt
README.md
```

## Repro rápido
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Opcional) Descargar modelo spaCy para POS
python -m spacy download es_core_news_lg

# Etiquetar POS y entrenar baseline
make pos tagger=spacy
make train cfg=experiments/configs/baseline.yaml
make eval cfg=experiments/configs/baseline.yaml
make plots
```

## Notas
- No subas audios o datos sensibles. Usa scripts en `/data/` para descargar datasets públicos.
- Si subes pesos grandes, usa Git LFS o publica como *Release assets*.


## Notebooks y flujo Colab → Código
- Usa `notebooks/` para experimentar y luego pasa el código limpio a `src/`.
- Listas para sintético en `data_generation_lists/`.
