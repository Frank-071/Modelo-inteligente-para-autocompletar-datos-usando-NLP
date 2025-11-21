# src/features/simple.py

def _token_features(tokens, i):
    """
    Features simples para el token i dentro de la oración tokens.
    """
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.len": len(word),
    }

    if i > 0:
        prev = tokens[i - 1]
        features.update({
            "-1:word.lower": prev.lower(),
            "-1:word.istitle": prev.istitle(),
            "-1:word.isupper": prev.isupper(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nex = tokens[i + 1]
        features.update({
            "+1:word.lower": nex.lower(),
            "+1:word.istitle": nex.istitle(),
            "+1:word.isupper": nex.isupper(),
        })
    else:
        features["EOS"] = True

    return features


def featurize(sentences):
    """
    sentences: lista de oraciones,
               cada oración = lista de (token, tag)
    Devuelve:
      X: lista de dicts de features
      y: lista de etiquetas BIO
    """
    X, y = [], []
    for sent in sentences:
        tokens = [w for (w, t) in sent]
        labels = [t for (w, t) in sent]
        for i in range(len(tokens)):
            X.append(_token_features(tokens, i))
            y.append(labels[i])
    return X, y
