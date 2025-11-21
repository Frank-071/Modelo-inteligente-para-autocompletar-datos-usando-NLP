# src/ner/svm_baseline.py
from typing import List, Tuple, Dict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import numpy as np
import os
import spacy
from spacy.tokens import Doc

# Carga spaCy (vectores + POS)
nlp = spacy.load("es_core_news_md", disable=["parser", "ner", "lemmatizer"])
VEC_DIM = nlp.vocab.vectors_length or 300

# Embedding context:
#   - "cur"  -> solo vector del token actual  (rápido)
#   - "pnc"  -> prev + cur + next            (más rico, más lento)
EMB_CTX = os.getenv("EMB_CTX", "cur").lower()  # "cur" | "pnc"
CTX_MULT = 1 if EMB_CTX == "cur" else 3

# Dimensión de la proyección aleatoria (a menor, más rápido/ligero)
PROJ_DIM = int(os.getenv("PROJ_DIM", "32"))

# Matriz de proyección aleatoria (fija para reproducibilidad)
_RND = np.random.RandomState(42)
_RP = _RND.normal(
    loc=0.0,
    scale=1.0/np.sqrt(CTX_MULT * VEC_DIM),
    size=(CTX_MULT * VEC_DIM, PROJ_DIM)
).astype("float32")

def _zero_vec():
    return np.zeros(VEC_DIM, dtype="float32")

def _shape(w: str) -> str:
    s = []
    for c in w:
        if c.isdigit(): s.append("d")
        elif c.isalpha(): s.append("X" if c.isupper() else "x")
        else: s.append(c)
    return "".join(s)

def token_features(sent: List[Tuple[str, str]], i: int) -> Dict[str, object]:
    w = sent[i][0]
    digs = "".join(ch for ch in w if ch.isdigit())
    feats: Dict[str, object] = {
        "w.lower": w.lower(),
        "w.isdigit": str(w.isdigit()),
        "w.isalpha": str(w.isalpha()),
        "w.istitle": str(w.istitle()),
        "w.shape": _shape(w),
        "pref1": w[:1], "pref2": w[:2], "pref3": w[:3],
        "suf1": w[-1:], "suf2": w[-2:], "suf3": w[-3:],
        # extras numéricos útiles
        "num.len": len(digs),
        "has.dash": "-" in w,
        "is.numlike": len(digs) > 0,
    }
    # char n-grams limitados (cap para no explotar features)
    wlow = w.lower()
    for n in (2, 3):
        if len(wlow) >= n:
            lim = min(5, len(wlow) - n + 1)
            for j in range(lim):
                feats[f"ch{n}={wlow[j:j+n]}"] = 1

    # contexto ±1 (simbólico)
    if i > 0:
        wprev = sent[i-1][0]
        feats.update({"-1.lower": wprev.lower(), "-1.shape": _shape(wprev)})
    else:
        feats["BOS"] = "1"
    if i < len(sent)-1:
        wnext = sent[i+1][0]
        feats.update({"+1.lower": wnext.lower(), "+1.shape": _shape(wnext)})
    else:
        feats["EOS"] = "1"
    return feats

def featurize(sents: List[List[Tuple[str,str]]]):
    """
    X: dict-features (rasgos + POS + embeddings comprimidos)
    y: etiquetas BIO
    Optimizado con nlp.pipe (batch, multiproceso).
    """
    X, y = [], []

    # Construye Docs respetando tu tokenización
    docs = [Doc(nlp.vocab, words=[w for (w, _) in sent]) for sent in sents]

    nproc = max(1, (os.cpu_count() or 2) - 1)
    for sent, doc in zip(sents, nlp.pipe(docs, batch_size=256, n_process=nproc)):
        if all(t.pos_ == "" for t in doc):
            if "tok2vec" in nlp.pipe_names:
                nlp.get_pipe("tok2vec")(doc)
            if "morphologizer" in nlp.pipe_names:
                nlp.get_pipe("morphologizer")(doc)
            elif "tagger" in nlp.pipe_names:
                nlp.get_pipe("tagger")(doc)
            else:
                raise RuntimeError("El modelo spaCy no tiene morphologizer ni tagger para POS.")

        # vectores por token (fallback a ceros)
        vecs = [(t.vector if t.has_vector else _zero_vec()) for t in doc]

        for i, (_, tag) in enumerate(sent):
            feats = token_features(sent, i)

            # POS
            feats["pos"] = doc[i].pos_

            # embeddings comprimidos
            if EMB_CTX == "cur":
                v = vecs[i]
            else:
                v_prev = vecs[i-1] if i > 0 else _zero_vec()
                v_cur  = vecs[i]
                v_next = vecs[i+1] if i < len(sent)-1 else _zero_vec()
                v = np.concatenate([v_prev, v_cur, v_next]).astype("float32")
            # si EMB_CTX=="cur", v ya es (VEC_DIM,); si "pnc", es (3*VEC_DIM,)
            if v.ndim == 1 and v.shape[0] != _RP.shape[0]:
                pad = np.zeros((_RP.shape[0] - v.shape[0],), dtype="float32")
                v = np.concatenate([v, pad])

            z = v @ _RP  # (PROJ_DIM,)
            for k in range(PROJ_DIM):
                feats[f"emb{k}"] = float(z[k])

            X.append(feats)
            y.append(tag)
    return X, y

def make_pipeline() -> Pipeline:
    # SVM con tolerancia/iteraciones algo relajadas para speed-up
    max_iter = int(os.getenv("SVM_MAXITER", "8000"))
    tol = float(os.getenv("SVM_TOL", "0.005"))
    return Pipeline([
        ("vec", DictVectorizer(sparse=True, dtype=np.float32)),
        ("clf", LinearSVC(
            C=0.5,
            max_iter=max_iter,
            tol=tol,
            dual=False,
            random_state=42
        ))
    ])



