# src/features/pos.py

from .simple import _token_features  # reutilizamos lo básico

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Ideal: instalar modelo mediano de español
            # python -m spacy download es_core_news_md
            try:
                _nlp = spacy.load("es_core_news_md")
            except Exception:
                _nlp = spacy.load("es_core_news_sm")
        except Exception:
            # Fallback ultra simple si no hay spaCy
            import spacy
            _nlp = spacy.blank("es")
    return _nlp


def _get_pos_tags(tokens):
    nlp = _get_nlp()
    text = " ".join(tokens)
    doc = nlp(text)

    # Si la segmentación no calza, hacemos un fallback token por token
    if len(doc) != len(tokens):
        tags = []
        for tok in tokens:
            d = nlp(tok)
            tags.append(d[0].pos_ if len(d) else "X")
        return tags

    return [t.pos_ for t in doc]


def featurize(sentences):
    X, y = [], []
    for sent in sentences:
        tokens = [w for (w, t) in sent]
        labels = [t for (w, t) in sent]
        pos_tags = _get_pos_tags(tokens)

        for i in range(len(tokens)):
            feats = _token_features(tokens, i)
            feats["pos"] = pos_tags[i] if i < len(pos_tags) else "X"
            X.append(feats)
            y.append(labels[i])
    return X, y
