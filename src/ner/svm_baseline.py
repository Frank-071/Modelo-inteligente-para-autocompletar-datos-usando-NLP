from typing import List, Tuple, Dict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

def _shape(w: str) -> str:
    # Forma simple de palabra (Aa9, etc.)
    s = []
    for c in w:
        if c.isdigit(): s.append("d")
        elif c.isalpha(): s.append("X" if c.isupper() else "x")
        else: s.append(c)
    return "".join(s)

def token_features(sent: List[Tuple[str, str]], i: int) -> Dict[str, str]:
    w = sent[i][0]
    feats = {
        "w.lower": w.lower(),
        "w.isdigit": str(w.isdigit()),
        "w.isalpha": str(w.isalpha()),
        "w.istitle": str(w.istitle()),
        "w.shape": _shape(w),
        "pref1": w[:1], "pref2": w[:2], "pref3": w[:3],
        "suf1": w[-1:], "suf2": w[-2:], "suf3": w[-3:],
    }
    # contexto ±1
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
    X, y = [], []
    for sent in sents:
        for i in range(len(sent)):
            X.append(token_features(sent, i))
            y.append(sent[i][1])  # tag BIO
    return X, y

def make_pipeline() -> Pipeline:
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LinearSVC(
            C=0.5,           # regulariza un poco más (ayuda a converger)
            max_iter=20000,  # más iteraciones
            tol=1e-3,        # tolerancia menos estricta
            dual=False,      # suele ir mejor si n_muestras > n_features
            random_state=42
            # class_weight="balanced",  # opcional si ves mucha desbalance (minoritarias vs 'O')
        ))
    ])
