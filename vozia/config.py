"""Configuração central do serviço de IA do Voz do Bairro.

Segredos (chave da OpenAI, chave do serviço) vêm SEMPRE de variáveis de ambiente.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "propostas.jsonl"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "docs" / "figures"

RANDOM_STATE = 42

CATEGORIAS = ["mobilidade", "zeladoria", "saúde", "lazer"]

# --- Modelos da OpenAI (podem ser trocados por variável de ambiente) ----------------
LLM_MODEL = os.getenv("VOZIA_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("VOZIA_EMBED_MODEL", "text-embedding-3-small")
STT_MODEL = os.getenv("VOZIA_STT_MODEL", "whisper-1")
LLM_TIMEOUT_S = float(os.getenv("VOZIA_LLM_TIMEOUT", "30"))

# --- Limites de entrada -------------------------------------------------------------
MAX_TEXTO = 2000            # caracteres por proposta ou comentário
MAX_COMENTARIOS = 50
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_EXTENSOES = {"mp3", "wav", "m4a", "webm", "ogg", "mp4", "mpeg", "mpga", "flac"}

# --- Detector de duplicadas (valores padrão; o evaluate.py grava os calibrados) ------
LIMIARES_PADRAO = {
    "tfidf": {"duplicada": 0.40, "relacionada": 0.25},
    "openai": {"duplicada": 0.62, "relacionada": 0.48},
}
LIMIAR_CATEGORIA = 0.50     # abaixo disso o app pede revisão humana da categoria
TOP_K = 5


def embedder_escolhido() -> str:
    """'openai' se houver chave; senão 'tfidf' (modo local, sem custo)."""
    pedido = os.getenv("VOZIA_EMBEDDER", "").strip().lower()
    if pedido in {"openai", "tfidf"}:
        return pedido
    return "openai" if os.getenv("OPENAI_API_KEY") else "tfidf"


def limiares(embedder: str) -> dict:
    """Limiares calibrados (models/<embedder>/limiares.json) ou os padrões."""
    arq = MODELS_DIR / embedder / "limiares.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    return dict(LIMIARES_PADRAO[embedder])
