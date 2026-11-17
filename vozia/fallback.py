"""Respostas determinísticas usadas quando o LLM está indisponível ou devolve algo inválido.

O fluxo do cidadão nunca trava por causa do LLM: ele só perde o texto 'bonito'.
"""
from __future__ import annotations

import re


def _primeira_frase(texto: str, limite: int) -> str:
    frase = re.split(r"(?<=[.!?])\s+", texto.strip())[0].rstrip(".!?")
    if len(frase) > limite:  # corta na última palavra inteira
        frase = frase[:limite].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return frase[:1].upper() + frase[1:]


def organizar(texto: str) -> dict:
    texto = " ".join(texto.split())
    return {"titulo": _primeira_frase(texto, 80), "descricao": texto[:600]}


def mensagem_duplicada(candidata: dict, confirmada: bool = True) -> dict:
    if confirmada:
        texto = (f"Parece que já existe uma proposta sobre isso no seu bairro: “{candidata['titulo']}” "
                 f"({candidata['apoios']} apoios). Que tal apoiá-la em vez de criar outra?")
    else:
        texto = (f"Encontramos uma proposta relacionada no seu bairro: “{candidata['titulo']}” "
                 f"({candidata['apoios']} apoios). Confira se não é o mesmo problema antes de publicar.")
    return {"mensagem": texto, "ids_citados": [candidata["id"]]}


def resumo_neutro(titulo: str, texto: str, comentarios: list[str]) -> dict:
    n = len(comentarios)
    corpo = f"Proposta: {titulo}. {_primeira_frase(texto, 200)}."
    corpo += f" A discussão tem {n} comentário(s)." if n else " Ainda não há comentários."
    return {"resumo": corpo, "pontos_a_favor": [], "preocupacoes": []}


def rascunho_parecer(titulo: str, apoios: int, decisao: str, justificativa: str, prazo_dias: int | None) -> dict:
    abertura = f"Prezados moradores, sobre a proposta “{titulo}”, que reuniu {apoios} apoios:"
    if decisao == "aceita":
        corpo = "a Prefeitura aceita a proposta e vai acompanhá-la publicamente até a conclusão."
    elif decisao == "agenda":
        quando = f" em até {prazo_dias} dias" if prazo_dias is not None else ""
        corpo = f"a Prefeitura agendou a execução{quando} e informará o andamento neste espaço."
    else:
        corpo = f"a Prefeitura não poderá atender no momento. Justificativa: {justificativa or 'a ser informada pelo gestor'}."
    return {"rascunho": f"{abertura} {corpo} (Rascunho gerado automaticamente: revise antes de publicar.)"}
