"""Detector de propostas duplicadas: recuperação por embeddings + decisão por limiares calibrados."""
from __future__ import annotations

from dataclasses import dataclass

from .store import Item, VectorStore


@dataclass
class Similar:
    item: Item
    score: float
    nivel: str  # "duplicada" | "relacionada" | "distinta"


def nivel_de(score: float, limiares: dict) -> str:
    if score >= limiares["duplicada"]:
        return "duplicada"
    if score >= limiares["relacionada"]:
        return "relacionada"
    return "distinta"


def buscar_similares(texto: str, bairro: str, store: VectorStore, embedder, limiares: dict,
                     k: int = 5, excluir_id: str | None = None) -> list[Similar]:
    vetor = embedder.embed([texto])[0]
    achados = store.buscar(vetor, bairro, k, excluir_id)
    return [Similar(it, s, nivel_de(s, limiares)) for it, s in achados]


def decisao(similares: list[Similar]) -> str:
    """'ja_existe' (sugerir apoiar), 'parecidas' (mostrar relacionadas) ou 'nova' (pode publicar)."""
    niveis = {s.nivel for s in similares}
    if "duplicada" in niveis:
        return "ja_existe"
    if "relacionada" in niveis:
        return "parecidas"
    return "nova"


def agrupar(store: VectorStore, bairro: str, limiar: float) -> list[list[str]]:
    """Agrupa propostas do bairro por ligação média (average linkage): dois grupos só se juntam se, em média,
    seus membros forem parecidos. Evita o 'efeito corrente' de ligar A-B-C só porque A~B e B~C."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    idx = [i for i, it in enumerate(store.itens) if it.bairro == bairro]
    if len(idx) < 2:
        return []
    sims = store.vetores[idx] @ store.vetores[idx].T
    dist = np.clip(1.0 - sims, 0.0, None)
    rotulos = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                      distance_threshold=1.0 - limiar).fit_predict(dist)
    grupos: dict[int, list[str]] = {}
    for pos, r in zip(idx, rotulos):
        grupos.setdefault(int(r), []).append(store.itens[pos].id)
    return sorted((g for g in grupos.values() if len(g) > 1), key=len, reverse=True)
