# src/main_eval.py
import argparse, os, json, joblib
from collections import Counter
from sklearn.metrics import classification_report, f1_score
from utils.conll import read_conll
from ner.svm_baseline import featurize

def main(test_path, model_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # 1) Cargar test y features
    test_sents = read_conll(test_path)
    Xt, yt = featurize(test_sents)

    # 2) Cargar modelo y predecir
    pipe = joblib.load(model_path)
    yhat = pipe.predict(Xt)
    assert len(yt) == len(yhat), "Desalineación: yt y yhat tienen longitudes distintas."

    # 3) Métrica token-level (incluye 'O')
    report_all = classification_report(yt, yhat, output_dict=True, zero_division=0)
    f1_macro_all = f1_score(yt, yhat, average="macro", zero_division=0)

    # 4) Métrica token-level NER (ignora 'O')
    keep = [i for i, y in enumerate(yt) if y != "O"]
    if keep:
        yt_ner  = [yt[i] for i in keep]
        yhat_ner = [yhat[i] for i in keep]
        report_ner = classification_report(yt_ner, yhat_ner, output_dict=True, zero_division=0)
        f1_macro_ner = f1_score(yt_ner, yhat_ner, average="macro", zero_division=0)
    else:
        report_ner, f1_macro_ner = {"_note": "No hay etiquetas distintas de 'O'."}, 0.0

    # 5) Guardar resultados
    with open(os.path.join(results_dir, "test_report_all.json"), "w", encoding="utf-8") as f:
        json.dump(report_all, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "test_report_ner.json"), "w", encoding="utf-8") as f:
        json.dump(report_ner, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "f1_macro_token_including_O": round(f1_macro_all, 6),
            "f1_macro_token_ner_only": round(f1_macro_ner, 6),
            "test_label_distribution": dict(Counter(yt))
        }, f, ensure_ascii=False, indent=2)

    # 6) Consola
    print(f"[OK] F1 token (incluye 'O'): {f1_macro_all:.4f}")
    print(f"[OK] F1 token NER (sin 'O'): {f1_macro_ner:.4f}")
    print(f"[OK] Resultados en: {results_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",  default="data/test.conll")
    ap.add_argument("--model", default="models/svm_baseline.pkl")
    ap.add_argument("--out",   default="experiments/results/svm_baseline")
    args = ap.parse_args()
    main(args.test, args.model, args.out)

