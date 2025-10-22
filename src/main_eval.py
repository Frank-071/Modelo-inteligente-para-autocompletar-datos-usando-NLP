import argparse, os, json, joblib
from sklearn.metrics import classification_report, f1_score
from utils.conll import read_conll
from ner.svm_baseline import featurize

def main(test_path, model_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    test_sents = read_conll(test_path)

    pipe = joblib.load(model_path)
    Xt, yt = featurize(test_sents)
    yhat = pipe.predict(Xt)

    report = classification_report(yt, yhat, output_dict=True, zero_division=0)
    f1_macro = f1_score(yt, yhat, average="macro", zero_division=0)

    with open(os.path.join(results_dir, "test_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"f1_macro": f1_macro}, f, ensure_ascii=False, indent=2)

    print(f"[OK] F1 macro (test): {f1_macro:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",  default="data/test.conll")
    ap.add_argument("--model", default="models/svm_baseline.pkl")
    ap.add_argument("--out",   default="experiments/results/svm_baseline")
    args = ap.parse_args()
    main(args.test, args.model, args.out)

