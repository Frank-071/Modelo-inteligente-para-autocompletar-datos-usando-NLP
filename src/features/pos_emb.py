# src/features/pos_emb.py

def featurize(sentences):
    """
    Aquí usamos la implementación avanzada de svm_baseline:
    rasgos + POS + embeddings proyectados.
    """
    from ner import svm_baseline as sb
    
    return sb.featurize(sentences)
