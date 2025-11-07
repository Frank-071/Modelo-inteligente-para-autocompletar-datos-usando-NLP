# src/main_eval.py

import argparse, os, json, joblib
from collections import Counter
from sklearn.metrics import classification_report, f1_score

from utils.conll import read_conll

from features import simple as feat_simple
from features import pos as feat_pos
from features import pos_emb as feat_pos_emb


# Debe matchear con main_train.py
EXPERIMENTS = {
    "exp1_svm_simple": {
        "feature_fn": feat_simple.featurize,
        "default_model_path": "models/exp1_svm_simple.pkl",
    },
    "exp2_svm_pos": {
        "feature_fn": feat_pos.featurize,
        "default_model_path": "models/exp2_svm_pos.pkl",
    },
    "exp3_svm_pos_emb_pro": {
        "feature_fn": feat_pos_emb.featurize,      # usa svm_baseline.featurize
        "default_model_path": "models/exp3_svm_pos_emb_pro.pkl",
    },
    "exp4_mlp_pos_emb_pro": {
        "feature_fn": feat_pos_emb.featurize,
        "default_model_path": "models/exp4_mlp_pos_emb_pro.pkl",
    },
}


def main(test_path, exp_name, model_path, results_dir):
    cfg = EXPERIMENTS[exp_name]
    feature_fn = cfg["feature_fn"]

    if model_path is None:
        model_path = cfg["default_model_path"]

    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Experimento: {exp_name}")
    print(f"[INFO] Cargando test: {test_path}")
    print(f"[INFO] Cargando modelo: {model_path}")

    # 1) Test + features
    test_sents = read_conll(test_path)
    Xt, yt = feature_fn(test_sents)

    # 2) Modelo
    model = joblib.load(model_path)
    yhat = model.predict(Xt)

    assert len(yt) == len(yhat), "Desalineación: yt y yhat tienen longitudes distintas."

    # 3) Métrica token-level (incluye 'O')
    report_all = classification_report(yt, yhat, output_dict=True, zero_division=0)
    f1_macro_all = f1_score(yt, yhat, average="macro", zero_division=0)

    # 4) Métrica token-level NER (ignora 'O')
    keep = [i for i, y in enumerate(yt) if y != "O"]
    if keep:
        yt_ner = [yt[i] for i in keep]
        yhat_ner = [yhat[i] for i in keep]
        report_ner = classification_report(yt_ner, yhat_ner, output_dict=True, zero_division=0)
        f1_macro_ner = f1_score(yt_ner, yhat_ner, average="macro", zero_division=0)
    else:
        report_ner, f1_macro_ner = {"_note": "No hay etiquetas distintas de 'O'."}, 0.0

    # 5) Guardar
    with open(os.path.join(results_dir, "test_report_all.json"), "w", encoding="utf-8") as f:
        json.dump(report_all, f, ensure_ascii=False, indent=2)

    with open(os.path.join(results_dir, "test_report_ner.json"), "w", encoding="utf-8") as f:
        json.dump(report_ner, f, ensure_ascii=False, indent=2)

    with open(os.path.join(results_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "experiment": exp_name,
            "model_path": model_path,
            "f1_macro_token_including_O": round(f1_macro_all, 6),
            "f1_macro_token_ner_only": round(f1_macro_ner, 6),
            "test_label_distribution": dict(Counter(yt)),
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] F1 token (incluye 'O'): {f1_macro_all:.4f}")
    print(f"[OK] F1 token NER (sin 'O'): {f1_macro_ner:.4f}")
    print(f"[OK] Resultados en: {results_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.conll")
    ap.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    ap.add_argument("--model", default=None, help="Opcional: ruta específica al .pkl")
    ap.add_argument("--out", required=True, help="Directorio para guardar resultados")
    args = ap.parse_args()

    main(args.test, args.exp, args.model, args.out)

