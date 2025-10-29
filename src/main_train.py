# src/main_train.py
# Asume corpus CONLL con etiquetas BIO (B-*, I-*, O)

import argparse, os, json, joblib
from sklearn.metrics import classification_report, f1_score
from utils.conll import read_conll
from ner.svm_baseline import featurize, make_pipeline

def main(train_path, val_path, model_path, results_dir):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_sents = read_conll(train_path)
    val_sents   = read_conll(val_path)

    Xtr, ytr = featurize(train_sents)
    Xva, yva = featurize(val_sents)

    pipe = make_pipeline()
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xva)
    report = classification_report(yva, yhat, output_dict=True, zero_division=0)
    f1_macro = f1_score(yva, yhat, average="macro", zero_division=0)

    joblib.dump(pipe, model_path)
    with open(os.path.join(results_dir, "val_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"f1_macro": f1_macro}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Modelo guardado en: {model_path}")
    print(f"[OK] F1 macro (val): {f1_macro:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.conll")
    ap.add_argument("--val",   default="data/val.conll")
    ap.add_argument("--model", default="models/svm_baseline.pkl")
    ap.add_argument("--out",   default="experiments/results/svm_baseline")
    args = ap.parse_args()
    main(args.train, args.val, args.model, args.out)
