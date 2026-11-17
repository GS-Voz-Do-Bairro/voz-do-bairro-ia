"""Contrato (JSON) do serviço. Limites de tamanho protegem contra abuso e custo."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from vozia import config

Texto = Field(..., min_length=10, max_length=config.MAX_TEXTO)
Bairro = Field(..., min_length=2, max_length=80)


class AnaliseEntrada(BaseModel):
    texto: str = Texto
    bairro: str = Bairro

    model_config = {"json_schema_extra": {"example": {
        "texto": "tem um buraco enorme na rua das flores perto da padaria, moto cai todo dia",
        "bairro": "Jardim das Acácias"}}}


class IndexarEntrada(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    bairro: str = Bairro
    titulo: str = Field(..., min_length=3, max_length=160)
    texto: str = Texto
    categoria: str = ""
    apoios: int = Field(0, ge=0)


class ResumoEntrada(BaseModel):
    titulo: str = Field(..., min_length=3, max_length=160)
    texto: str = Texto
    comentarios: list[str] = Field(default_factory=list, max_length=config.MAX_COMENTARIOS)


class ParecerEntrada(BaseModel):
    titulo: str = Field(..., min_length=3, max_length=160)
    texto: str = Texto
    resumo: str = Field("", max_length=1200)
    apoios: int = Field(..., ge=0)
    decisao: Literal["aceita", "recusa", "agenda"]
    justificativa: str = Field("", max_length=1000)
    prazo_dias: Optional[int] = Field(None, ge=1, le=3650)


class GruposEntrada(BaseModel):
    bairro: str = Bairro
