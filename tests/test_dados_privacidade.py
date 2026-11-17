from vozia.bootstrap import carregar_dataset
from vozia.privacy import mascarar_pii


def test_dataset_estrutura():
    ds = carregar_dataset()
    assert len(ds) == 144
    por_issue = {}
    for d in ds:
        por_issue.setdefault(d["issue_id"], []).append(d)
    assert len(por_issue) == 36 and all(len(v) == 4 for v in por_issue.values())
    assert {d["categoria"] for d in ds} == {"mobilidade", "zeladoria", "saúde", "lazer"}
    # um problema pertence a um único bairro e uma única categoria
    assert all(len({d["bairro"] for d in v}) == 1 and len({d["categoria"] for d in v}) == 1 for v in por_issue.values())


def test_dataset_sem_dados_pessoais():
    for d in carregar_dataset():
        assert mascarar_pii(d["texto"])[1] == 0, d["texto"]


def test_mascara_cpf_telefone_email():
    texto = "Meu CPF é 123.456.789-09, ligue (11) 91234-5678 ou escreva para maria@exemplo.com.br"
    limpo, n = mascarar_pii(texto)
    assert "123.456.789-09" not in limpo and "91234-5678" not in limpo and "maria@" not in limpo
    assert "[CPF]" in limpo and "[TELEFONE]" in limpo and "[EMAIL]" in limpo and n == 3


def test_texto_sem_pii_nao_muda():
    t = "A linha 305 passa só de hora em hora e tem 40 pessoas esperando"
    assert mascarar_pii(t) == (t, 0)
