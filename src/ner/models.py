# src/ner/models.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


def make_svm_simple():
    """
    SVM lineal clásico para NER con features de tipo diccionario.
    """
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LinearSVC())
    ])


def make_mlp():
    """
    MLP sencillo para comparar contra SVM usando las mismas features.
    """
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            max_iter=50,
            verbose=False,
        ))
    ])
