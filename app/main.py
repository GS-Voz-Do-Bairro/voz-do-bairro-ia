"""API do serviço de IA do Voz do Bairro (consumida pela API Java).

Segurança: se AI_SERVICE_API_KEY estiver definida (obrigatório em produção), todas as rotas, exceto
/health, exigem o cabeçalho X-API-Key. Nenhuma rota devolve ou registra a chave da OpenAI.
"""
from __future__ import annotations

import hmac
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import APIKeyHeader

from vozia import bootstrap, config
from vozia.store import Item
from vozia.transcribe import AudioInvalido, TranscricaoIndisponivel

from .schemas import AnaliseEntrada, GruposEntrada, IndexarEntrada, ParecerEntrada, ResumoEntrada

app = FastAPI(
    title="Voz do Bairro IA",
    version="1.0.0",
    description="IA da plataforma CitizenTech: detecta propostas duplicadas (RAG), classifica a categoria, "
                "transcreve áudio e gera resumos e rascunhos de parecer. Apoio à decisão: quem decide é a pessoa.",
)


@lru_cache(maxsize=1)
def servico():
    return bootstrap.carregar()


_esquema_chave = APIKeyHeader(name="X-API-Key", auto_error=False)  # faz o Swagger mostrar o botão Authorize


def exigir_chave(x_api_key: str | None = Depends(_esquema_chave)) -> None:
    esperada = os.getenv("AI_SERVICE_API_KEY")
    if esperada and not hmac.compare_digest(x_api_key or "", esperada):
        raise HTTPException(status_code=401, detail="Chave de API ausente ou inválida")


Auth = Depends(exigir_chave)


@app.get("/health", tags=["infra"])
def health():
    s = servico()
    return {"status": "ok", "embedder": s.embedder.nome, "propostas_indexadas": len(s.store),
            "llm_configurado": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/propostas/analisar", dependencies=[Auth], tags=["propostas"])
def analisar(entrada: AnaliseEntrada):
    """Antes de publicar: organiza o texto, sugere a categoria e avisa se já existe proposta parecida."""
    return servico().analisar(entrada.texto, entrada.bairro)


@app.get("/demo", include_in_schema=False)
def demo():
    from pathlib import Path
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent / "static" / "demo.html")


@app.post("/audio/propostas", dependencies=[Auth], tags=["propostas"])
async def analisar_audio(arquivo: UploadFile = File(...), bairro: str = Form(..., min_length=2, max_length=80)):
    """Proposta por áudio: transcreve (Whisper), organiza e detecta duplicadas."""
    conteudo = await arquivo.read(config.MAX_AUDIO_BYTES + 1)
    try:
        return servico().analisar_audio(conteudo, arquivo.filename or "audio", bairro)
    except AudioInvalido as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TranscricaoIndisponivel:
        raise HTTPException(status_code=503, detail="Transcrição indisponível no momento; envie a proposta em texto.")


@app.post("/propostas/indexar", dependencies=[Auth], status_code=201, tags=["índice"])
def indexar(entrada: IndexarEntrada):
    """Chamada pela API Java quando uma proposta é publicada (ou tem os apoios atualizados)."""
    item = servico().indexar(Item(**entrada.model_dump()))
    return {"id": item.id, "indexada": True}


@app.delete("/propostas/{proposta_id}", dependencies=[Auth], tags=["índice"])
def remover(proposta_id: str):
    if not servico().remover(proposta_id):
        raise HTTPException(status_code=404, detail="Proposta não está no índice")
    return {"id": proposta_id, "removida": True}


@app.post("/propostas/grupos", dependencies=[Auth], tags=["índice"])
def grupos(entrada: GruposEntrada):
    """Agrupa propostas parecidas do bairro (para o gestor ver os clusters)."""
    return {"bairro": entrada.bairro, "grupos": servico().grupos(entrada.bairro)}


@app.post("/propostas/resumo", dependencies=[Auth], tags=["gestor"])
def resumo(entrada: ResumoEntrada):
    """Resumo neutro da proposta e da discussão."""
    return servico().resumir(entrada.titulo, entrada.texto, entrada.comentarios)


@app.post("/pareceres/rascunho", dependencies=[Auth], tags=["gestor"])
def parecer(entrada: ParecerEntrada):
    """Rascunho de parecer para o gestor revisar: a decisão (aceita, recusa, agenda) é sempre dele."""
    return servico().parecer(entrada.titulo, entrada.texto, entrada.resumo, entrada.apoios,
                             entrada.decisao, entrada.justificativa, entrada.prazo_dias)
