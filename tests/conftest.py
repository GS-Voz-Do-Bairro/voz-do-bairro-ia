import json
from types import SimpleNamespace

import pytest

from vozia import bootstrap, config
from vozia.llm import LLM
from vozia.servico import VozService
from vozia.transcribe import Transcritor


class FakeOpenAI:
    """Cliente falso da OpenAI: devolve respostas programadas e guarda o que foi enviado (sem rede, sem chave)."""

    def __init__(self, chat=None, texto_audio="tem um buraco enorme na rua das flores", falha=False):
        self.respostas = list(chat or [])
        self.enviados = []
        self.falha = falha
        self.texto_audio = texto_audio
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._audio))

    def _chat(self, **kw):
        self.enviados.append(kw)
        if self.falha:
            raise ConnectionError("sem rede")
        bruto = self.respostas.pop(0) if self.respostas else "{}"
        bruto = bruto if isinstance(bruto, str) else json.dumps(bruto, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=bruto))])

    def _audio(self, **kw):
        if self.falha:
            raise ConnectionError("sem rede")
        return SimpleNamespace(text=self.texto_audio)


@pytest.fixture(scope="session")
def base():
    """Embedder TF-IDF + índice + classificador construídos do dataset (sem gravar em disco)."""
    return bootstrap.construir("tfidf", salvar=False)


@pytest.fixture
def fabrica_servico(base):
    def _f(client=None, texto_audio="tem um buraco enorme na rua das flores"):
        client = client or FakeOpenAI(texto_audio=texto_audio)
        embedder, store, clf = base
        # cópia do índice para um teste não afetar os outros
        from vozia.store import Item, VectorStore
        loja = VectorStore()
        for it, v in zip(store.itens, store.vetores):
            loja.upsert(Item(**it.__dict__), v)
        svc = VozService(embedder, loja, clf, LLM(client), Transcritor(client), config.limiares("tfidf"))
        return svc, client
    return _f
