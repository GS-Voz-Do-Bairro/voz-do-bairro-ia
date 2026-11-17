"""Índice vetorial em memória com persistência em disco (RAG básico: recuperar propostas parecidas).

Para o volume de uma plataforma de bairro (milhares de propostas) busca exata por produto escalar é
suficiente e transparente. Em escala maior, troque por pgvector/Qdrant mantendo esta interface.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Item:
    id: str
    bairro: str
    titulo: str
    texto: str
    categoria: str = ""
    apoios: int = 0


class VectorStore:
    def __init__(self) -> None:
        self.itens: list[Item] = []
        self.vetores = np.zeros((0, 0))
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.itens)

    def _pos(self, item_id: str) -> int | None:
        for i, it in enumerate(self.itens):
            if it.id == item_id:
                return i
        return None

    def upsert(self, item: Item, vetor: np.ndarray) -> None:
        with self._lock:
            vetor = np.asarray(vetor, dtype=float).reshape(1, -1)
            i = self._pos(item.id)
            if i is not None:
                self.itens[i] = item
                self.vetores[i] = vetor[0]
            elif len(self.itens) == 0:
                self.itens, self.vetores = [item], vetor
            else:
                self.itens.append(item)
                self.vetores = np.vstack([self.vetores, vetor])

    def remover(self, item_id: str) -> bool:
        with self._lock:
            i = self._pos(item_id)
            if i is None:
                return False
            del self.itens[i]
            self.vetores = np.delete(self.vetores, i, axis=0)
            return True

    def buscar(self, vetor: np.ndarray, bairro: str | None, k: int, excluir_id: str | None = None):
        """Top-k itens mais parecidos DO MESMO BAIRRO: [(Item, similaridade de cosseno)]."""
        with self._lock:
            if not self.itens:
                return []
            sims = self.vetores @ np.asarray(vetor, dtype=float).reshape(-1)
            ordem = np.argsort(-sims)
            saida = []
            for i in ordem:
                it = self.itens[i]
                if (bairro and it.bairro != bairro) or it.id == excluir_id:
                    continue
                saida.append((it, float(sims[i])))
                if len(saida) == k:
                    break
            return saida

    def salvar(self, pasta: Path) -> None:
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "itens.json").write_text(
            json.dumps([asdict(i) for i in self.itens], ensure_ascii=False), encoding="utf-8")
        np.save(pasta / "vetores.npy", self.vetores)

    @classmethod
    def carregar(cls, pasta: Path) -> "VectorStore":
        s = cls()
        s.itens = [Item(**d) for d in json.loads((pasta / "itens.json").read_text(encoding="utf-8"))]
        s.vetores = np.load(pasta / "vetores.npy")
        return s
