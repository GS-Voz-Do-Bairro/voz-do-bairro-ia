"""Transcrição de áudio (Whisper, OpenAI) com validação de formato e tamanho."""
from __future__ import annotations

import os

from . import config


class AudioInvalido(ValueError):
    """Formato, tamanho ou conteúdo do áudio não permite transcrever (erro do cliente)."""


class TranscricaoIndisponivel(RuntimeError):
    """O serviço de transcrição falhou (erro do servidor/provedor)."""


class Transcritor:
    def __init__(self, client=None, modelo: str | None = None) -> None:
        self._client = client
        self.modelo = modelo or config.STT_MODEL

    @property
    def client(self):
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise TranscricaoIndisponivel("OPENAI_API_KEY não definida")
            from openai import OpenAI

            self._client = OpenAI(timeout=60, max_retries=1)
        return self._client

    def transcrever(self, conteudo: bytes, nome_arquivo: str) -> str:
        ext = nome_arquivo.rsplit(".", 1)[-1].lower() if "." in nome_arquivo else ""
        if ext not in config.AUDIO_EXTENSOES:
            raise AudioInvalido(f"formato '{ext or '?'}' não suportado; use {', '.join(sorted(config.AUDIO_EXTENSOES))}")
        if not conteudo:
            raise AudioInvalido("arquivo de áudio vazio")
        if len(conteudo) > config.MAX_AUDIO_BYTES:
            raise AudioInvalido(f"áudio maior que {config.MAX_AUDIO_BYTES // (1024 * 1024)} MB")
        try:
            resp = self.client.audio.transcriptions.create(
                model=self.modelo, file=(nome_arquivo, conteudo), language="pt")
            texto = (resp.text if hasattr(resp, "text") else str(resp)).strip()
        except TranscricaoIndisponivel:
            raise
        except Exception as exc:
            raise TranscricaoIndisponivel(f"falha na transcrição: {type(exc).__name__}") from exc
        if len(texto) < 8:
            raise AudioInvalido("não foi possível reconhecer fala no áudio")
        return texto[:config.MAX_TEXTO]
