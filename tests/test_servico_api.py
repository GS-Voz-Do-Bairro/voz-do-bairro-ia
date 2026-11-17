import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeOpenAI
from vozia.embeddings import EmbedderErro

BAIRRO = "Jardim das Acácias"
TEXTO_DUP = "buraco na rua das flores perto da padaria, uma moto caiu ontem"
ORG = {"titulo": "Buraco na Rua das Flores", "descricao": "Há um buraco na Rua das Flores, perto da padaria."}
SIM = {"vereditos": [{"id": "m01-1", "mesmo_problema": True}]}
NAO = {"vereditos": [{"id": "m01-1", "mesmo_problema": False}]}
MSG = {"mensagem": "Já existe uma proposta parecida: Buraco na Rua das Flores. Quer apoiar?", "ids_citados": ["m01-1"]}


def _svc(fabrica, respostas, **kw):
    c = FakeOpenAI(respostas)
    svc, _ = fabrica(c, **kw)
    return svc, c


def test_fluxo_completo_com_llm(fabrica_servico):
    svc, _ = _svc(fabrica_servico, [ORG, SIM, MSG])
    r = svc.analisar(TEXTO_DUP, BAIRRO)
    assert r["modo"] == "llm" and r["verificacao"] == "llm" and r["decisao"] == "ja_existe"
    assert r["similares"][0]["id"] == "m01-1" and r["categoria"]["categoria"] == "mobilidade"
    assert r["mensagem_ao_cidadao"]["ids_citados"] == ["m01-1"] and r["organizado"]["titulo"].startswith("Buraco")


def test_juiz_rebaixa_falso_alarme_para_parecida(fabrica_servico):
    msg = dict(MSG, ids_citados=["m01-1"])
    svc, c = _svc(fabrica_servico, [ORG, NAO, msg])
    r = svc.analisar(TEXTO_DUP, BAIRRO)
    assert r["decisao"] == "parecidas" and r["similares"][0]["nivel"] == "relacionada"
    assert "RELACIONADAS" in c.enviados[-1]["messages"][0]["content"]       # tom cauteloso: não afirma que é igual


def test_sem_llm_usa_fallback_e_continua(fabrica_servico):
    svc, _ = fabrica_servico(FakeOpenAI(falha=True))
    r = svc.analisar(TEXTO_DUP, BAIRRO)
    assert r["modo"] == "fallback" and r["verificacao"] == "limiar"
    assert r["organizado"]["titulo"] and r["decisao"] in ("ja_existe", "parecidas")
    assert r["mensagem_ao_cidadao"]["ids_citados"]


def test_fallback_nao_afirma_igualdade_quando_so_relacionada():
    from vozia import fallback
    cand = {"id": "a", "titulo": "Buraco", "apoios": 3}
    assert "já existe" in fallback.mensagem_duplicada(cand, True)["mensagem"]
    assert "relacionada" in fallback.mensagem_duplicada(cand, False)["mensagem"]


def test_pii_nao_chega_ao_llm(fabrica_servico):
    svc, c = _svc(fabrica_servico, [ORG, SIM, MSG])
    r = svc.analisar("buraco na rua das flores, falar com João no (11) 91234-5678 ou CPF 123.456.789-09", BAIRRO)
    enviado = " ".join(m["content"] for kw in c.enviados for m in kw["messages"])
    assert "91234-5678" not in enviado and "123.456.789-09" not in enviado
    assert r["pii_mascarados"] == 2 and "[CPF]" in r["texto_limpo"]


def test_falha_no_embedder_nao_bloqueia(fabrica_servico):
    svc, _ = _svc(fabrica_servico, [ORG])

    class Quebrado:
        nome = "quebrado"

        def embed(self, t):
            raise EmbedderErro("fora do ar")
    svc.embedder = Quebrado()
    r = svc.analisar(TEXTO_DUP, BAIRRO)
    assert r["decisao"] == "nao_verificado" and "indisponível" in r["aviso"] and r["mensagem_ao_cidadao"] is None


def test_proposta_nova_nao_gera_alerta(fabrica_servico):
    svc, _ = _svc(fabrica_servico, [ORG])
    r = svc.analisar("Quero aulas gratuitas de xadrez e dama para os aposentados jogarem", BAIRRO)
    assert r["decisao"] != "ja_existe"


def test_indexar_novo_item_e_detectar_depois(fabrica_servico):
    from vozia.store import Item
    svc, _ = fabrica_servico(FakeOpenAI(falha=True))
    svc.indexar(Item("n1", BAIRRO, "Aulas de xadrez", "Aulas gratuitas de xadrez para aposentados na praça. CPF 123.456.789-09", "lazer", 4))
    assert "123.456.789-09" not in svc.store.itens[-1].texto               # nada pessoal no índice
    r = svc.analisar("aulas de xadrez gratuitas para aposentados na praça", BAIRRO)
    assert r["similares"][0]["id"] == "n1"


def test_grupos_e_remocao(fabrica_servico):
    from vozia.store import Item
    svc, _ = fabrica_servico()
    svc.indexar(Item("m01-9", BAIRRO, "Buraco", "buraco na rua das flores perto da padaria, moto cai", "mobilidade", 1))
    assert any({"m01-1", "m01-9"} <= set(g) for g in svc.grupos(BAIRRO))
    assert svc.remover("m01-9") and not svc.remover("m01-9")


def test_resumo_e_parecer_com_fallback(fabrica_servico):
    svc, _ = fabrica_servico(FakeOpenAI(falha=True))
    r = svc.resumir("Buraco na rua", "Há um buraco na Rua das Flores.", ["apoio", "concordo"])
    assert r["modo"] == "fallback" and "2 comentário" in r["resumo"]
    p = svc.parecer("Buraco na rua", "texto", "", 40, "recusa", "sem orçamento neste ano", None)
    assert "sem orçamento" in p["rascunho"] and "Rascunho" in p["aviso"] and p["modo"] == "fallback"


# --------------------------------------------------------------------- API HTTP
@pytest.fixture
def cliente(fabrica_servico, monkeypatch):
    from app import main
    svc, c = fabrica_servico(FakeOpenAI([ORG, SIM, MSG]))
    monkeypatch.setattr(main, "servico", lambda: svc)
    monkeypatch.delenv("AI_SERVICE_API_KEY", raising=False)
    return TestClient(main.app), c


def test_health_sem_chave(cliente, monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "segredo")
    r = cliente[0].get("/health")
    assert r.status_code == 200 and r.json()["embedder"] == "tfidf"


def test_autenticacao(cliente, monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "segredo")
    corpo = {"texto": TEXTO_DUP, "bairro": BAIRRO}
    assert cliente[0].post("/propostas/analisar", json=corpo).status_code == 401
    assert cliente[0].post("/propostas/analisar", json=corpo, headers={"X-API-Key": "errada"}).status_code == 401
    assert cliente[0].post("/propostas/analisar", json=corpo, headers={"X-API-Key": "segredo"}).status_code == 200


def test_analisar_http(cliente):
    r = cliente[0].post("/propostas/analisar", json={"texto": TEXTO_DUP, "bairro": BAIRRO})
    assert r.status_code == 200 and r.json()["decisao"] == "ja_existe"


@pytest.mark.parametrize("corpo", [
    {"texto": "curto", "bairro": BAIRRO},
    {"texto": "x" * 2001, "bairro": BAIRRO},
    {"texto": TEXTO_DUP, "bairro": ""},
    {"texto": TEXTO_DUP},
])
def test_entrada_invalida_422(cliente, corpo):
    assert cliente[0].post("/propostas/analisar", json=corpo).status_code == 422


def test_audio_ok_e_invalidos(cliente):
    c, _ = cliente
    ok = c.post("/audio/propostas", files={"arquivo": ("voz.mp3", b"fake-audio", "audio/mpeg")}, data={"bairro": BAIRRO})
    assert ok.status_code == 200 and "buraco" in ok.json()["transcricao"]
    ruim = c.post("/audio/propostas", files={"arquivo": ("voz.exe", b"x", "application/octet-stream")}, data={"bairro": BAIRRO})
    assert ruim.status_code == 400
    vazio = c.post("/audio/propostas", files={"arquivo": ("voz.mp3", b"", "audio/mpeg")}, data={"bairro": BAIRRO})
    assert vazio.status_code == 400


def test_audio_servico_fora_do_ar_503(fabrica_servico, monkeypatch):
    from app import main
    svc, _ = fabrica_servico(FakeOpenAI(falha=True))
    monkeypatch.setattr(main, "servico", lambda: svc)
    monkeypatch.delenv("AI_SERVICE_API_KEY", raising=False)
    r = TestClient(main.app).post("/audio/propostas", files={"arquivo": ("v.mp3", b"abc", "audio/mpeg")}, data={"bairro": BAIRRO})
    assert r.status_code == 503 and "texto" in r.json()["detail"]


def test_indexar_remover_grupos_http(cliente):
    c, _ = cliente
    corpo = {"id": "p-1", "bairro": BAIRRO, "titulo": "Nova coisa", "texto": "Texto de uma proposta nova para o bairro", "apoios": 2}
    assert c.post("/propostas/indexar", json=corpo).status_code == 201
    assert c.delete("/propostas/p-1").status_code == 200
    assert c.delete("/propostas/p-1").status_code == 404
    assert c.post("/propostas/grupos", json={"bairro": BAIRRO}).status_code == 200


def test_parecer_decisao_invalida_422(cliente):
    base = {"titulo": "Buraco", "texto": "Buraco na rua das flores", "apoios": 10, "decisao": "talvez"}
    assert cliente[0].post("/pareceres/rascunho", json=base).status_code == 422
    assert cliente[0].post("/pareceres/rascunho", json=dict(base, decisao="agenda", prazo_dias=0)).status_code == 422
