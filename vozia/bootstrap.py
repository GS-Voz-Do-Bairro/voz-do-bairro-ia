"""Monta (ou carrega) o serviço: embedder, índice de propostas publicadas e classificador de categoria."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config
from .categoria import ClassificadorCategoria
from .embeddings import OpenAIEmbedder, TfidfEmbedder
from .fallback import organizar
from .llm import LLM
from .servico import VozService
from .store import Item, VectorStore
from .transcribe import Transcritor


def carregar_dataset(caminho: Path = config.DATA_PATH) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _apoios_demo(item_id: str) -> int:
    """Apoios fictícios e estáveis, só para a demonstração."""
    return 3 + sum(map(ord, item_id)) % 118


def construir(nome: str, dataset: list[dict] | None = None, salvar: bool = True):
    """Treina o classificador e indexa 1 proposta por problema (as demais variantes servem de consulta)."""
    dataset = dataset or carregar_dataset()
    textos = [d["texto"] for d in dataset]
    embedder = TfidfEmbedder().fit(textos) if nome == "tfidf" else OpenAIEmbedder()
    clf = ClassificadorCategoria(embedder).fit(textos, [d["categoria"] for d in dataset])

    store = VectorStore()
    publicadas = [d for d in dataset if d["id"].endswith("-1")]
    vetores = embedder.embed([d["texto"] for d in publicadas])
    for d, v in zip(publicadas, vetores):
        store.upsert(Item(d["id"], d["bairro"], organizar(d["texto"])["titulo"], d["texto"],
                          d["categoria"], _apoios_demo(d["id"])), v)

    if salvar:
        pasta = config.MODELS_DIR / nome
        pasta.mkdir(parents=True, exist_ok=True)
        if nome == "tfidf":
            embedder.salvar(pasta / "embedder.joblib")
        clf.salvar(pasta / "categoria.joblib")
        store.salvar(pasta / "indice")
    return embedder, store, clf


def carregar(nome: str | None = None) -> VozService:
    nome = nome or config.embedder_escolhido()
    pasta = config.MODELS_DIR / nome
    try:
        if nome == "tfidf":
            embedder = TfidfEmbedder.carregar(pasta / "embedder.joblib")
        else:
            embedder = OpenAIEmbedder()
        clf = ClassificadorCategoria.carregar(embedder, pasta / "categoria.joblib")
        store = VectorStore.carregar(pasta / "indice")
    except Exception:  # artefato ausente ou gerado com outra versão do scikit-learn: reconstrói
        embedder, store, clf = construir(nome)
    return VozService(embedder, store, clf, LLM(), Transcritor(), config.limiares(nome))
