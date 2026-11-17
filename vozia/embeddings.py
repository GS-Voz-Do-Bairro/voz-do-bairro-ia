"""Embeddings: OpenAI (produção) ou TF-IDF de caracteres (local, sem custo, usado nos testes e como baseline).

Os dois implementam a mesma interface, então o resto do código não sabe qual está em uso.
Todos os vetores saem normalizados (norma 1): similaridade de cosseno = produto escalar.
"""
from __future__ import annotations

import os
from typing import Protocol

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config


class EmbedderErro(RuntimeError):
    """Falha ao gerar embeddings (rede, chave inválida, limite de uso)."""


class Embedder(Protocol):
    nome: str

    def embed(self, textos: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Baseline lexical: TF-IDF de n-gramas de caracteres (tolera erros de digitação e acentos)."""

    nome = "tfidf"

    def __init__(self) -> None:
        self.vetorizador = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            strip_accents="unicode", lowercase=True, max_features=8192,
        )
        self._ajustado = False

    def fit(self, textos: list[str]) -> "TfidfEmbedder":
        self.vetorizador.fit(textos)
        self._ajustado = True
        return self

    def embed(self, textos: list[str]) -> np.ndarray:
        if not self._ajustado:
            raise EmbedderErro("TfidfEmbedder precisa de fit() antes de embed()")
        return normalize(self.vetorizador.transform(textos).toarray())

    def salvar(self, caminho) -> None:
        joblib.dump(self.vetorizador, caminho)

    @classmethod
    def carregar(cls, caminho) -> "TfidfEmbedder":
        e = cls()
        e.vetorizador = joblib.load(caminho)
        e._ajustado = True
        return e


class OpenAIEmbedder:
    """Embeddings semânticos da OpenAI (text-embedding-3-small por padrão)."""

    nome = "openai"

    def __init__(self, client=None, modelo: str | None = None) -> None:
        if client is None:
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise EmbedderErro("OPENAI_API_KEY não definida")
            client = OpenAI(timeout=config.LLM_TIMEOUT_S, max_retries=1)
        self.client = client
        self.modelo = modelo or config.EMBED_MODEL

    def embed(self, textos: list[str]) -> np.ndarray:
        vetores: list[list[float]] = []
        try:
            for i in range(0, len(textos), 100):
                resp = self.client.embeddings.create(model=self.modelo, input=textos[i:i + 100])
                vetores.extend(d.embedding for d in resp.data)
        except Exception as exc:  # rede, chave, cota
            raise EmbedderErro(f"falha ao gerar embeddings: {type(exc).__name__}") from exc
        return normalize(np.asarray(vetores, dtype=float))
