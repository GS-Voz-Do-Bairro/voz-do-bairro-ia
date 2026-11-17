"""Classificador de categoria (mobilidade, zeladoria, saúde, lazer): regressão logística sobre embeddings."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config


class ClassificadorCategoria:
    def __init__(self, embedder, clf: LogisticRegression | None = None) -> None:
        self.embedder = embedder
        self.clf = clf

    def fit(self, textos: list[str], rotulos: list[str]) -> "ClassificadorCategoria":
        X = self.embedder.embed(textos)
        self.clf = LogisticRegression(max_iter=2000, C=10.0, random_state=config.RANDOM_STATE)
        self.clf.fit(X, rotulos)
        return self

    def prever(self, texto: str) -> dict:
        probs = self.clf.predict_proba(self.embedder.embed([texto]))[0]
        ordem = np.argsort(-probs)
        cats = self.clf.classes_
        conf = float(probs[ordem[0]])
        return {
            "categoria": str(cats[ordem[0]]),
            "confianca": round(conf, 3),
            "alternativas": [{"categoria": str(cats[i]), "confianca": round(float(probs[i]), 3)} for i in ordem[1:3]],
            "precisa_revisao": conf < config.LIMIAR_CATEGORIA,
        }

    def salvar(self, caminho) -> None:
        joblib.dump(self.clf, caminho)

    @classmethod
    def carregar(cls, embedder, caminho) -> "ClassificadorCategoria":
        return cls(embedder, joblib.load(caminho))
