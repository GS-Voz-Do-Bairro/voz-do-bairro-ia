import numpy as np
import pytest

from vozia.embeddings import EmbedderErro, OpenAIEmbedder, TfidfEmbedder
from vozia.similar import agrupar, buscar_similares, decisao, nivel_de
from vozia.store import Item, VectorStore

LIM = {"duplicada": 0.40, "relacionada": 0.20}


def test_tfidf_vetores_normalizados(base):
    e = base[0]
    v = e.embed(["buraco na rua", "semáforo quebrado"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_tfidf_exige_fit():
    with pytest.raises(EmbedderErro):
        TfidfEmbedder().embed(["x"])


def test_openai_embedder_normaliza_e_encapsula_erro():
    class C:
        def __init__(self, falha=False):
            self.falha = falha
            self.embeddings = self

        def create(self, model, input):
            if self.falha:
                raise TimeoutError("lento")
            from types import SimpleNamespace
            return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])
    v = OpenAIEmbedder(client=C()).embed(["a", "b"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    with pytest.raises(EmbedderErro):
        OpenAIEmbedder(client=C(falha=True)).embed(["a"])


def test_store_busca_por_bairro_e_exclusao():
    s = VectorStore()
    s.upsert(Item("a", "X", "t", "x"), np.array([1.0, 0.0]))
    s.upsert(Item("b", "Y", "t", "x"), np.array([1.0, 0.0]))
    s.upsert(Item("c", "X", "t", "x"), np.array([0.0, 1.0]))
    assert [i.id for i, _ in s.buscar(np.array([1.0, 0.0]), "X", 5)] == ["a", "c"]   # b é de outro bairro
    assert [i.id for i, _ in s.buscar(np.array([1.0, 0.0]), "X", 5, excluir_id="a")] == ["c"]
    assert s.remover("a") and not s.remover("a") and len(s) == 2


def test_store_upsert_substitui_e_persiste(tmp_path):
    s = VectorStore()
    s.upsert(Item("a", "X", "t1", "x", apoios=1), np.array([1.0, 0.0]))
    s.upsert(Item("a", "X", "t2", "x", apoios=9), np.array([0.0, 1.0]))
    assert len(s) == 1 and s.itens[0].apoios == 9
    s.salvar(tmp_path)
    r = VectorStore.carregar(tmp_path)
    assert r.itens[0].titulo == "t2" and np.allclose(r.vetores[0], [0.0, 1.0])


def test_niveis_e_decisao():
    assert nivel_de(0.5, LIM) == "duplicada" and nivel_de(0.3, LIM) == "relacionada" and nivel_de(0.1, LIM) == "distinta"
    from vozia.similar import Similar
    mk = lambda n: Similar(Item("i", "X", "t", "x"), 0.0, n)
    assert decisao([mk("distinta"), mk("duplicada")]) == "ja_existe"
    assert decisao([mk("relacionada"), mk("distinta")]) == "parecidas"
    assert decisao([mk("distinta")]) == "nova" and decisao([]) == "nova"


def test_parafrase_recupera_o_original_no_mesmo_bairro(base):
    e, store, _ = base
    r = buscar_similares("buraco na rua das flores, perto da padaria, derrubou uma moto", "Jardim das Acácias",
                         store, e, LIM, k=3)
    assert r[0].item.id == "m01-1"


def test_nao_recupera_de_outro_bairro(base):
    e, store, _ = base
    r = buscar_similares("buraco na rua das flores", "Centro", store, e, LIM, k=10)
    assert all(s.item.bairro == "Centro" for s in r)


def test_agrupar_junta_variantes():
    s = VectorStore()
    s.upsert(Item("a", "X", "t", "x"), np.array([1.0, 0.0]))
    s.upsert(Item("b", "X", "t", "x"), np.array([0.99, 0.14]) / np.linalg.norm([0.99, 0.14]))
    s.upsert(Item("c", "X", "t", "x"), np.array([0.0, 1.0]))
    assert agrupar(s, "X", 0.9) == [["a", "b"]]
