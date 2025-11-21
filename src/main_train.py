
# src/main_train.py
# Entrenamiento NER BIO para varios experimentos.

import argparse
import os
import json
import joblib

from sklearn.metrics import classification_report, f1_score

from utils.conll import read_conll

from ner import svm_baseline as sb
from ner.models import make_svm_simple, make_mlp
from features import simple as feat_simple
from features import pos as feat_pos
from features import pos_emb as feat_pos_emb


EXPERIMENTS = {
    # Exp 1: solo rasgos simples + SVM clásico
    "exp1_svm_simple": {
        "feature_fn": feat_simple.featurize,
        "model_fn": make_svm_simple,
        "default_model_path": "models/exp1_svm_simple.pkl",
        "description": "SVM con features simples (forma + contexto).",
    },
    # Exp 2: rasgos simples + POS + SVM clásico
    "exp2_svm_pos": {
        "feature_fn": feat_pos.featurize,
        "model_fn": make_svm_simple,
        "default_model_path": "models/exp2_svm_pos.pkl",
        "description": "SVM con features simples + POS.",
    },
    # Exp 3: versión PRO: rasgos + POS + embeddings proyectados + SVM avanzado
    "exp3_svm_pos_emb_pro": {
        "feature_fn": feat_pos_emb.featurize,
        "model_fn": sb.make_pipeline,
        "default_model_path": "models/exp3_svm_pos_emb_pro.pkl",
        "description": "SVM PRO con simples + POS + embeddings proyectados.",
    },
    # Exp 4: mismos features PRO pero modelo MLP
    "exp4_mlp_pos_emb_pro": {
        "feature_fn": feat_pos_emb.featurize,
        "model_fn": make_mlp,
        "default_model_path": "models/exp4_mlp_pos_emb_pro.pkl",
        "description": "MLP con simples + POS + embeddings proyectados.",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Entrenamiento NER BIO para distintos experimentos."
    )
    p.add_argument("--train", required=True, help="Ruta a train.conll")
    p.add_argument("--val", required=True, help="Ruta a val.conll")
    p.add_argument(
        "--exp",
        required=True,
        choices=EXPERIMENTS.keys(),
        help="Nombre del experimento a ejecutar.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Ruta para guardar el modelo (.pkl). Si no se da, usa la default del experimento.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Directorio donde guardar métricas y reportes.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = EXPERIMENTS[args.exp]

    feature_fn = cfg["feature_fn"]
    model_fn = cfg["model_fn"]

    model_path = args.model if args.model else cfg["default_model_path"]
    out_dir = args.out

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Experimento: {args.exp}")
    print(f"[INFO] {cfg['description']}")
    print(f"[INFO] Cargando train: {args.train}")
    print(f"[INFO] Cargando val: {args.val}")

    train_sents = read_conll(args.train)
    val_sents = read_conll(args.val)

    print(f"[INFO] Generando features (train)...")
    X_train, y_train = feature_fn(train_sents)

    print(f"[INFO] Generando features (val)...")
    X_val, y_val = feature_fn(val_sents)

    print(f"[INFO] Creando modelo...")
    model = model_fn()

    print(f"[INFO] Entrenando...")
    model.fit(X_train, y_train)

    print(f"[INFO] Evaluando en validación...")
    y_pred = model.predict(X_val)

    f1_macro = f1_score(y_val, y_pred, average="macro")
    report_dict = classification_report(y_val, y_pred, output_dict=True)
    report_txt = classification_report(y_val, y_pred)

    print(f"[INFO] F1-macro (val): {f1_macro:.4f}")

    # Guardar modelo
    joblib.dump(model, model_path)
    print(f"[INFO] Modelo guardado en {model_path}")

    # Guardar métricas
    metrics = {
        "experiment": args.exp,
        "description": cfg["description"],
        "f1_macro_val": f1_macro,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"[INFO] Resultados guardados en {out_dir}")
    print("[INFO] Listo.")


if __name__ == "__main__":
    main()

