import os
import json
import matplotlib.pyplot as plt

# Configura acá tus experimentos y carpetas
EXPERIMENTS = [
    ("Exp1 SVM simple",           "exp1_svm_simple"),
    ("Exp2 SVM + POS",            "exp2_svm_pos"),
    ("Exp3 SVM PRO + Emb",        "exp3_svm_pos_emb_pro"),
    ("Exp4 MLP + Emb",            "exp4_mlp_pos_emb_pro"),
]

BASE_DIR = "experiments/results"
OUT_DIR = "experiments/plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load_val_f1(exp_dir):
    path = os.path.join(BASE_DIR, exp_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return m.get("f1_macro_val")


def load_test_f1(exp_dir):
    path = os.path.join(BASE_DIR, f"{exp_dir.replace('exp', 'exp')}_test", "test_metrics.json")
    # ej: exp1_svm_simple -> exp1_test (ajusta si tus nombres difieren)
    if not os.path.exists(path):
        # fallback: asume carpeta expX_test
        path = os.path.join(BASE_DIR, f"{exp_dir.split('_')[0]}_test", "test_metrics.json")
        if not os.path.exists(path):
            return None, None
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return (
        m.get("f1_macro_token_including_O"),
        m.get("f1_macro_token_ner_only"),
    )


def plot_val_f1():
    labels, vals = [], []
    for name, d in EXPERIMENTS:
        v = load_val_f1(d)
        if v is not None:
            labels.append(name)
            vals.append(v)

    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("F1-macro (val)")
    plt.xticks(rotation=15, ha="right")
    plt.title("Resultados en validación por experimento")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "val_f1_macro.png"))


def plot_test_f1():
    labels, f_all, f_ner = [], [], []
    for name, d in EXPERIMENTS:
        fa, fn = load_test_f1(d)
        if fa is not None:
            labels.append(name)
            f_all.append(fa)
            f_ner.append(fn)

    # F1-macro (incluye O)
    plt.figure()
    plt.bar(labels, f_all)
    plt.ylabel("F1-macro (test)")
    plt.xticks(rotation=15, ha="right")
    plt.title("Resultados en prueba (incluye 'O')")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "test_f1_macro.png"))

    # F1-macro NER (sin O)
    plt.figure()
    plt.bar(labels, f_ner)
    plt.ylabel("F1-macro NER (test, sin 'O')")
    plt.xticks(rotation=15, ha="right")
    plt.title("Resultados en prueba (solo entidades)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "test_f1_ner.png"))


if __name__ == "__main__":
    plot_val_f1()
    plot_test_f1()
    print("[INFO] Gráficos guardados en", OUT_DIR)