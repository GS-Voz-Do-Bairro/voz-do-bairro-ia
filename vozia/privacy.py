"""Proteção de dados pessoais: mascara CPF, CNPJ, telefone, e-mail e RG antes de enviar texto a um LLM.

Limite assumido: nomes de pessoas e endereços completos não são detectados por expressões regulares.
Por isso o serviço também NÃO envia o id nem os dados cadastrais do autor ao LLM.
"""
from __future__ import annotations

import re

_PADROES = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("TELEFONE", re.compile(r"(?<!\d)(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-\s]?\d{4}(?!\d)")),
    ("RG", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX]\b")),
]


def mascarar_pii(texto: str) -> tuple[str, int]:
    """Devolve (texto com [CPF], [TELEFONE]... no lugar dos dados, quantidade de trechos mascarados)."""
    total = 0
    for rotulo, padrao in _PADROES:
        texto, n = padrao.subn(f"[{rotulo}]", texto)
        total += n
    return texto, total
