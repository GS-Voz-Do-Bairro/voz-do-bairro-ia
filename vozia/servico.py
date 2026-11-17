"""Orquestra o fluxo de IA: PII -> organizar (LLM) -> categoria (ML) -> similares (RAG) -> mensagem (LLM)."""
from __future__ import annotations

from dataclasses import asdict

from . import config, fallback
from .categoria import ClassificadorCategoria
from .embeddings import EmbedderErro
from .llm import LLM, LLMIndisponivel
from .privacy import mascarar_pii
from .similar import agrupar, buscar_similares, decisao
from .store import Item, VectorStore
from .transcribe import Transcritor


class VozService:
    def __init__(self, embedder, store: VectorStore, classificador: ClassificadorCategoria | None,
                 llm: LLM, transcritor: Transcritor, limiares: dict) -> None:
        self.embedder, self.store, self.classificador = embedder, store, classificador
        self.llm, self.transcritor, self.limiares = llm, transcritor, limiares

    # -------------------------------------------------------------- fluxo principal
    def analisar(self, texto: str, bairro: str) -> dict:
        limpo, n_pii = mascarar_pii(texto)
        modo = "llm"

        try:
            org = self.llm.organizar(limpo).model_dump()
        except LLMIndisponivel:
            org, modo = fallback.organizar(limpo), "fallback"

        categoria = self.classificador.prever(limpo) if self.classificador else None

        similares, aviso, verificacao = [], None, "limiar"
        try:
            similares = buscar_similares(limpo, bairro, self.store, self.embedder, self.limiares, config.TOP_K)
        except EmbedderErro:
            aviso = "Detector de propostas parecidas indisponível; publique com atenção."

        candidatas = [s for s in similares if s.nivel != "distinta"][:3]
        cand = [{"id": s.item.id, "titulo": s.item.titulo, "texto": s.item.texto, "apoios": s.item.apoios}
                for s in candidatas]
        if cand:  # etapa 2 do RAG: o LLM confirma se é mesmo o mesmo problema
            try:
                veredito = self.llm.verificar_duplicadas(limpo, cand)
                for s in candidatas:
                    s.nivel = "duplicada" if veredito[s.item.id] else "relacionada"
                verificacao = "llm"
            except LLMIndisponivel:
                modo = "fallback"  # sem LLM, vale o limiar calibrado
        decide = "nao_verificado" if aviso else decisao(similares)

        mensagem = None
        confirmadas = [c for s, c in zip(candidatas, cand) if s.nivel == "duplicada"]
        if cand and decide in ("ja_existe", "parecidas"):
            citadas, certa = (confirmadas, True) if confirmadas else (cand[:1], False)
            try:
                mensagem = self.llm.mensagem_duplicada(limpo, citadas, certa).model_dump()
            except LLMIndisponivel:
                mensagem, modo = fallback.mensagem_duplicada(citadas[0], certa), "fallback"

        return {
            "texto_limpo": limpo, "pii_mascarados": n_pii, "organizado": org, "categoria": categoria,
            "decisao": decide, "verificacao": verificacao, "mensagem_ao_cidadao": mensagem, "aviso": aviso,
            "modo": modo,
            "similares": [{"id": s.item.id, "titulo": s.item.titulo, "apoios": s.item.apoios,
                           "similaridade": round(s.score, 3), "nivel": s.nivel} for s in similares],
            "embedder": getattr(self.embedder, "nome", "?"),
        }

    def analisar_audio(self, conteudo: bytes, nome_arquivo: str, bairro: str) -> dict:
        transcricao = self.transcritor.transcrever(conteudo, nome_arquivo)
        resultado = self.analisar(transcricao, bairro)
        resultado["transcricao"] = resultado["texto_limpo"]
        return resultado

    # -------------------------------------------------------------- índice
    def indexar(self, item: Item) -> Item:
        item.titulo, _ = mascarar_pii(item.titulo)
        item.texto, _ = mascarar_pii(item.texto)
        self.store.upsert(item, self.embedder.embed([item.texto])[0])
        return item

    def remover(self, item_id: str) -> bool:
        return self.store.remover(item_id)

    def grupos(self, bairro: str) -> list[list[str]]:
        return agrupar(self.store, bairro, self.limiares["duplicada"])

    # -------------------------------------------------------------- IA para o gestor
    def resumir(self, titulo: str, texto: str, comentarios: list[str]) -> dict:
        titulo, _ = mascarar_pii(titulo)
        texto, _ = mascarar_pii(texto)
        coment = [mascarar_pii(c)[0] for c in comentarios]
        try:
            return {**self.llm.resumo_neutro(titulo, texto, coment).model_dump(), "modo": "llm"}
        except LLMIndisponivel:
            return {**fallback.resumo_neutro(titulo, texto, coment), "modo": "fallback"}

    def parecer(self, titulo: str, texto: str, resumo: str, apoios: int, decisao_gestor: str,
                justificativa: str, prazo_dias: int | None) -> dict:
        titulo, _ = mascarar_pii(titulo)
        texto, _ = mascarar_pii(texto)
        try:
            r = self.llm.rascunho_parecer(titulo, texto, resumo, apoios, decisao_gestor, justificativa, prazo_dias)
            return {**r.model_dump(), "modo": "llm", "aviso": "Rascunho: o gestor revisa e assina."}
        except LLMIndisponivel:
            return {**fallback.rascunho_parecer(titulo, apoios, decisao_gestor, justificativa, prazo_dias),
                    "modo": "fallback", "aviso": "Rascunho: o gestor revisa e assina."}
