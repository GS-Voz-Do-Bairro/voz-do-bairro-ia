"""Avaliação do detector de duplicadas (RAG) e do classificador de categoria.

Uso:  python -m vozia.evaluate                      # baselines locais (sem custo, sem chave)
      OPENAI_API_KEY=... python -m vozia.evaluate --openai   # inclui embeddings da OpenAI

Protocolo (evita vazamento):
- Os problemas (issues) são divididos em DEV e TESTE (metade de cada categoria). Os limiares são escolhidos
  só com pares do DEV; todos os números reportados vêm do TESTE.
- Embedders lexicais são ajustados (fit) só com textos do DEV.
- Cenário "duplicada": a consulta é uma proposta de um problema que JÁ está no índice (com as outras variantes).
- Cenário "nova": o índice não contém nenhuma variante do problema da consulta (falso alarme = erro).
- Classificador: validação cruzada agrupada por problema (nenhum problema aparece em treino e teste).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from . import config
from .bootstrap import carregar_dataset
from .llm import LLM
from .servico import VozService
from .transcribe import Transcritor
from .bootstrap import construir
from .embeddings import OpenAIEmbedder, TfidfEmbedder


class PalavrasEmbedder:
    """Baseline mais simples: TF-IDF de palavras (1-2 gramas), sem tolerância a erro de digitação."""

    nome = "palavras"

    def __init__(self):
        self.v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")

    def fit(self, textos):
        self.v.fit(textos)
        return self

    def embed(self, textos):
        return normalize(self.v.transform(textos).toarray())


def dividir_issues(dataset):
    por_cat = defaultdict(list)
    for d in dataset:
        if d["issue_id"] not in por_cat[d["categoria"]]:
            por_cat[d["categoria"]].append(d["issue_id"])
    teste = {i for ids in por_cat.values() for k, i in enumerate(sorted(ids)) if k % 2 == 1}
    dev = {d["issue_id"] for d in dataset} - teste
    return dev, teste


def _melhor_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def _limiar_relacionada(y, s, dup):
    """Menor limiar (<= duplicada) que ainda captura 95% das duplicadas do DEV."""
    pos = np.sort(s[y == 1])
    return float(min(dup, pos[int(0.05 * len(pos))]))


def avaliar_duplicadas(nome, dataset, embedder, ajustar, llm=None):
    dev, teste = dividir_issues(dataset)
    textos = [d["texto"] for d in dataset]
    if ajustar:
        embedder.fit([d["texto"] for d in dataset if d["issue_id"] in dev])
    E = embedder.embed(textos)
    S = E @ E.T
    issue = np.array([d["issue_id"] for d in dataset])
    bairro = np.array([d["bairro"] for d in dataset])
    n = len(dataset)

    def pares(conj):
        ij = [(i, j) for i, j in combinations(range(n), 2)
              if issue[i] in conj and issue[j] in conj and bairro[i] == bairro[j]]
        y = np.array([int(issue[i] == issue[j]) for i, j in ij])
        s = np.array([S[i, j] for i, j in ij])
        return ij, y, s

    _, y_dev, s_dev = pares(dev)
    ij_te, y_te, s_te = pares(teste)
    lim_dup, f1_dev = _melhor_f1(y_dev, s_dev)
    lim_rel = _limiar_relacionada(y_dev, s_dev, lim_dup)
    pred = (s_te >= lim_dup).astype(int)

    # --- recuperação como no produto: consulta contra o índice do mesmo bairro
    hit1 = hit3 = rr = n_dup = acerto_dup = n_nova = alarme = parecidas_nova = 0
    erros = []
    for q in range(n):
        if issue[q] not in teste:
            continue
        mesmo = [i for i in range(n) if i != q and bairro[i] == bairro[q]]
        ordem = sorted(mesmo, key=lambda i: -S[q, i])
        topo = ordem[0]
        n_dup += 1
        hit1 += issue[topo] == issue[q]
        hit3 += any(issue[i] == issue[q] for i in ordem[:3])
        rr += next(1 / (k + 1) for k, i in enumerate(ordem) if issue[i] == issue[q])
        acerto_dup += S[q, topo] >= lim_dup and issue[topo] == issue[q]
        # cenário "nova": remove do índice todas as variantes do problema
        sem = [i for i in mesmo if issue[i] != issue[q]]
        melhor = max(sem, key=lambda i: S[q, i])
        n_nova += 1
        if S[q, melhor] >= lim_dup:
            alarme += 1
            erros.append((dataset[q]["texto"], dataset[melhor]["texto"], round(float(S[q, melhor]), 3)))
        elif S[q, melhor] >= lim_rel:
            parecidas_nova += 1

    agrup = _avaliar_agrupamento(S, issue, bairro, teste, lim_dup)
    juiz = _avaliar_juiz(dataset, S, issue, bairro, teste, lim_rel, llm) if llm else {}
    m = {
        **juiz, "agrupamento": agrup, "embedder": nome, "limiares": {"duplicada": round(lim_dup, 3), "relacionada": round(lim_rel, 3)},
        "pares_teste": int(len(y_te)), "pares_duplicados_teste": int(y_te.sum()),
        "auc_roc": round(float(roc_auc_score(y_te, s_te)), 3),
        "pr_auc": round(float(average_precision_score(y_te, s_te)), 3),
        "precisao": round(float(precision_score(y_te, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_te, pred)), 3),
        "f1": round(float(f1_score(y_te, pred)), 3),
        "hit@1": round(hit1 / n_dup, 3), "hit@3": round(hit3 / n_dup, 3), "mrr": round(rr / n_dup, 3),
        "cenario_duplicada_ja_existe_correto": round(acerto_dup / n_dup, 3),
        "cenario_nova_falso_alarme": round(alarme / n_nova, 3),
        "cenario_nova_marcada_parecida": round(parecidas_nova / n_nova, 3),
        "consultas_teste": n_dup,
    }
    return m, erros[:5], (y_te, s_te)


def _avaliar_agrupamento(S, issue, bairro, teste, limiar):
    """Agrupa as propostas do TESTE por bairro (ligação média, mesmo limiar de 'duplicada') e compara com os problemas reais."""
    verdade, previsto, ofs = [], [], 0
    for b in sorted(set(bairro)):
        ix = [i for i in range(len(issue)) if bairro[i] == b and issue[i] in teste]
        if len(ix) < 2:
            continue
        D = np.clip(1.0 - S[np.ix_(ix, ix)], 0.0, None)
        r = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                    distance_threshold=1.0 - limiar).fit_predict(D)
        verdade += [issue[i] for i in ix]
        previsto += [int(x) + ofs for x in r]
        ofs += int(r.max()) + 1
    return {"propostas": len(verdade), "problemas_reais": len(set(verdade)), "grupos_encontrados": len(set(previsto)),
            "ari": round(float(adjusted_rand_score(verdade, previsto)), 3),
            "homogeneidade": round(float(homogeneity_score(verdade, previsto)), 3),
            "completude": round(float(completeness_score(verdade, previsto)), 3)}


def _avaliar_juiz(dataset, S, issue, bairro, teste, lim_rel, llm):
    """Etapa 2 do RAG (LLM confirma 'mesmo problema'), medida nos mesmos dois cenários do TESTE."""
    from .fallback import organizar

    def cand(i):
        return {"id": dataset[i]["id"], "titulo": organizar(dataset[i]["texto"])["titulo"],
                "texto": dataset[i]["texto"], "apoios": 0}

    n = dup_ok = nova_alarme = verif = verif_ok = 0
    for q in range(len(dataset)):
        if issue[q] not in teste:
            continue
        n += 1
        mesmo = [i for i in range(len(dataset)) if i != q and bairro[i] == bairro[q]]
        for cenario, pool in (("dup", mesmo), ("nova", [i for i in mesmo if issue[i] != issue[q]])):
            top = [i for i in sorted(pool, key=lambda i: -S[q, i])[:3] if S[q, i] >= lim_rel]
            if not top:
                continue
            try:
                v = llm.verificar_duplicadas(dataset[q]["texto"], [cand(i) for i in top])
            except Exception:
                continue
            confirmados = [i for i in top if v[dataset[i]["id"]]]
            verif += len(top)
            verif_ok += sum(bool(v[dataset[i]["id"]]) == (issue[i] == issue[q]) for i in top)
            if cenario == "dup":
                dup_ok += any(issue[i] == issue[q] for i in confirmados)
            else:
                nova_alarme += bool(confirmados)
    return {"juiz_llm": {"consultas": n, "candidatas_julgadas": verif,
                         "acuracia_do_veredito": round(verif_ok / max(verif, 1), 3),
                         "cenario_duplicada_ja_existe_correto": round(dup_ok / n, 3),
                         "cenario_nova_falso_alarme": round(nova_alarme / n, 3)}}


class _ClienteSemRede:
    """Cliente que sempre falha: força o caminho SEM LLM (fallback), mesmo com OPENAI_API_KEY definida."""

    def __init__(self):
        from types import SimpleNamespace

        def _falha(**_):
            raise ConnectionError("LLM desativado nesta medição")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_falha))


def avaliar_aceitacao(nome, servico):
    """Fluxo COMPLETO do serviço (analisar) com 40 propostas inéditas: o que o cidadão realmente veria."""
    casos = carregar_dataset(config.ROOT / "data" / "aceitacao.jsonl")
    contagem = defaultdict(lambda: defaultdict(int))
    falhas = []
    for c in casos:
        r = servico.analisar(c["texto"], c["bairro"])
        topo = r["similares"][0]["id"].split("-")[0] if r["similares"] else None
        if c["tipo"] == "duplicada":
            ok = r["decisao"] == "ja_existe" and any(
                s["nivel"] == "duplicada" and s["id"].split("-")[0] == c["issue_esperado"] for s in r["similares"])
            contagem["duplicada"]["ok" if ok else "erro"] += 1
            contagem["duplicada"]["top1_certo"] += int(topo == c["issue_esperado"])
        else:
            ok = r["decisao"] != "ja_existe"
            contagem[c["tipo"]]["ok" if ok else "erro"] += 1
        if not ok:
            falhas.append((c["tipo"], c["texto"], r["decisao"], r["similares"][0]["titulo"] if r["similares"] else ""))
    n = lambda t: sum(contagem[t][k] for k in ("ok", "erro"))
    d, nf, nd = contagem["duplicada"], contagem["nova_facil"], contagem["nova_dificil"]
    acertos = d["ok"] + nf["ok"] + nd["ok"]
    return {"configuracao": nome, "casos": len(casos),
            "acuracia_geral": round(acertos / len(casos), 3),
            "duplicadas_detectadas": f"{d['ok']}/{n('duplicada')}", "duplicadas_top1_certo": f"{d['top1_certo']}/{n('duplicada')}",
            "novas_faceis_sem_alarme": f"{nf['ok']}/{n('nova_facil')}", "novas_dificeis_sem_alarme": f"{nd['ok']}/{n('nova_dificil')}",
            "recall_duplicadas": round(d["ok"] / n("duplicada"), 3),
            "precisao_ja_existe": round(d["ok"] / max(d["ok"] + nf["erro"] + nd["erro"], 1), 3),
            "falhas": falhas[:8]}


def avaliar_categoria(nome, dataset, fabrica):
    textos = np.array([d["texto"] for d in dataset])
    y = np.array([d["categoria"] for d in dataset])
    grupos = np.array([d["issue_id"] for d in dataset])
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(textos, y, grupos):
        emb = fabrica()
        if hasattr(emb, "fit"):
            emb.fit(list(textos[tr]))
        clf = LogisticRegression(max_iter=2000, C=10.0, random_state=config.RANDOM_STATE)
        clf.fit(emb.embed(list(textos[tr])), y[tr])
        pred[te] = clf.predict(emb.embed(list(textos[te])))
    maioria = np.full_like(y, max(set(y), key=list(y).count))
    cats = config.CATEGORIAS
    return {
        "embedder": nome, "acuracia": round(float(accuracy_score(y, pred)), 3),
        "f1_macro": round(float(f1_score(y, pred, average="macro")), 3),
        "baseline_maioria_acuracia": round(float(accuracy_score(y, maioria)), 3),
        "categorias": cats, "matriz_confusao": confusion_matrix(y, pred, labels=cats).tolist(),
        "erros": [(str(textos[i]), str(y[i]), str(pred[i])) for i in np.where(pred != y)[0][:6]],
    }


def figuras(resultados):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    for nome, (y, s) in resultados.items():
        p, r, _ = precision_recall_curve(y, s)
        eixos[0].plot(r, p, label=nome)
    eixos[0].set(xlabel="Recall", ylabel="Precisão", title="Duplicadas: curva precisão x recall (teste)")
    eixos[0].legend()
    nome = "tfidf" if "tfidf" in resultados else next(iter(resultados))
    y, s = resultados[nome]
    eixos[1].hist(s[y == 0], bins=30, alpha=0.7, label="problemas diferentes", color="#9FCFC9")
    eixos[1].hist(s[y == 1], bins=30, alpha=0.8, label="mesmo problema", color="#00A896")
    eixos[1].set(xlabel="Similaridade de cosseno", ylabel="Pares", title=f"Distribuição das similaridades ({nome})")
    eixos[1].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "duplicadas.png", dpi=130)
    plt.close(fig)


def escrever_relatorio(dup, cat, erros, rodou_openai, aceit=()):
    L = ["# Avaliação da IA do Voz do Bairro", "",
         "> Dataset **sintético** (144 propostas escritas à mão, 36 problemas, 4 bairros). Os números medem o pipeline "
         "e a técnica; não são desempenho em produção.", "",
         "Protocolo: problemas divididos em DEV (calibra limiares) e TESTE (todos os números abaixo); embedders "
         "lexicais ajustados só no DEV; classificador com validação cruzada agrupada por problema.", "",
         "## 1. Detector de duplicadas (recuperação por embeddings)", "",
         "| Embedder | AUC-ROC | PR-AUC | Precisão | Recall | F1 | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|---|---|---|---|---|"]
    for m in dup:
        L.append(f"| {m['embedder']} | {m['auc_roc']} | {m['pr_auc']} | {m['precisao']} | {m['recall']} | {m['f1']} | "
                 f"{m['hit@1']} | {m['hit@3']} | {m['mrr']} |")
    L += ["", f"Pares de teste: {dup[0]['pares_teste']} ({dup[0]['pares_duplicados_teste']} do mesmo problema). "
              f"Consultas de teste: {dup[0]['consultas_teste']}.", "",
          "### Cenários de uso (consulta contra o índice do bairro)", "",
          "| Embedder | Limiar duplicada | Duplicada detectada (top-1 certo) | Nova com falso alarme | Nova marcada só como parecida |",
          "|---|---|---|---|---|"]
    for m in dup:
        L.append(f"| {m['embedder']} | {m['limiares']['duplicada']} | {m['cenario_duplicada_ja_existe_correto']} | "
                 f"{m['cenario_nova_falso_alarme']} | {m['cenario_nova_marcada_parecida']} |")
    for m in dup:
        if "juiz_llm" in m:
            j = m["juiz_llm"]
            L += ["", f"### Etapa 2: LLM confirma o 'mesmo problema' ({m['embedder']} + GPT)", "",
                  f"- Acurácia do veredito por candidata: **{j['acuracia_do_veredito']}** ({j['candidatas_julgadas']} julgadas)",
                  f"- Duplicada confirmada: **{j['cenario_duplicada_ja_existe_correto']}**",
                  f"- Falso alarme em proposta nova: **{j['cenario_nova_falso_alarme']}**"]
    L += ["", "### Agrupamento de propostas parecidas (ligação média, mesmo limiar de 'duplicada', só TESTE)", "",
          "| Embedder | Propostas | Problemas reais | Grupos encontrados | ARI | Homogeneidade | Completude |", "|---|---|---|---|---|---|---|"]
    for m in dup:
        a = m["agrupamento"]
        L.append(f"| {m['embedder']} | {a['propostas']} | {a['problemas_reais']} | {a['grupos_encontrados']} | "
                 f"{a['ari']} | {a['homogeneidade']} | {a['completude']} |")
    if aceit:
        L += ["", "### Teste de aceitação: fluxo completo com 40 propostas inéditas (`data/aceitacao.jsonl`)", "",
              "20 duplicadas com redação nova, 10 novas de assunto diferente e 10 novas *difíceis* (mesma rua/equipamento, problema diferente).", "",
              "| Configuração | Acurácia | Duplicadas detectadas | Top-1 certo | Novas fáceis sem alarme | Novas difíceis sem alarme |",
              "|---|---|---|---|---|---|"]
        for a in aceit:
            L.append(f"| {a['configuracao']} | {a['acuracia_geral']} | {a['duplicadas_detectadas']} | {a['duplicadas_top1_certo']} | "
                     f"{a['novas_faceis_sem_alarme']} | {a['novas_dificeis_sem_alarme']} |")
        for a in aceit:
            if a["falhas"]:
                L += ["", f"Falhas ({a['configuracao']}):"]
                L += [f"- [{t}] “{x}” → {d}; mais parecida: “{ti}”" for t, x, d, ti in a["falhas"]]
    if not rodou_openai:
        L += ["", "> **Embeddings da OpenAI e verificação por LLM ainda não medidos neste relatório.** Rode "
                  "`python -m vozia.evaluate --openai` (usa a chave e gera custo de centavos) e o relatório é regerado "
                  "com essas linhas."]
    L += ["", "![Duplicadas](../docs/figures/duplicadas.png)", "", "### Falsos alarmes (exemplos)", ""]
    for nome, lista in erros.items():
        for a, b, sc in lista[:3]:
            L.append(f"- ({nome}, similaridade {sc}) “{a}” × “{b}”")
    L += ["", "## 2. Classificador de categoria (validação cruzada agrupada por problema)", "",
          "| Embedder | Acurácia | F1 macro | Baseline (classe majoritária) |", "|---|---|---|---|"]
    for c in cat:
        L.append(f"| {c['embedder']} | {c['acuracia']} | {c['f1_macro']} | {c['baseline_maioria_acuracia']} |")
    L += ["", "Embeddings lexicais acertam pouco porque a categoria é **semântica**: um problema novo usa palavras que o "
              "treino nunca viu. Embeddings semânticos são o caminho esperado para essa tarefa; por segurança o serviço "
              "só *sugere* a categoria e pede revisão quando a confiança é baixa (< 0,50).", "",
          "## 3. Situações de falha testadas (pytest)", "",
          "LLM fora do ar, resposta do LLM inválida, id ou número inventado pelo LLM, tentativa de prompt injection, "
          "embeddings fora do ar, áudio inválido/vazio/grande/sem fala, entrada acima do limite, chave de API errada, "
          "dados pessoais no texto. Veja `tests/`.", "",
          "## 4. Limitações", "",
          "- Poucos problemas (36) e textos curtos escritos pela equipe: faltam gírias, textos longos e erros reais de áudio.",
          "- Todas as consultas de cada problema vêm do mesmo autor do dataset: tende a ser mais fácil que o mundo real.",
          "- O LLM pode errar o veredito; por isso a decisão final de publicar é sempre do cidadão e o resumo/parecer são rascunhos.",
          "- Nomes de pessoas e endereços completos não são detectados pelo mascaramento de dados pessoais."]
    (config.REPORTS_DIR / "avaliacao.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai", action="store_true", help="inclui embeddings da OpenAI (usa a chave e gera custo)")
    args = ap.parse_args()
    dataset = carregar_dataset()
    candidatos = [("palavras", PalavrasEmbedder, True), ("tfidf", TfidfEmbedder, True)]
    if args.openai:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("defina OPENAI_API_KEY para usar --openai")
        candidatos.append(("openai", OpenAIEmbedder, False))

    dup, cat, curvas, erros = [], [], {}, {}
    for nome, fabrica, ajustar in candidatos:
        juiz = None
        if nome == "openai":
            juiz = LLM()
        m, err, curva = avaliar_duplicadas(nome, dataset, fabrica(), ajustar, juiz)
        dup.append(m)
        curvas[nome], erros[nome] = curva, err
        cat.append(avaliar_categoria(nome, dataset, fabrica))
        if nome in ("tfidf", "openai"):  # limiares usados pelo serviço
            pasta = config.MODELS_DIR / nome
            pasta.mkdir(parents=True, exist_ok=True)
            (pasta / "limiares.json").write_text(json.dumps(m["limiares"]), encoding="utf-8")

    aceit = []
    for nome in ["tfidf"] + (["openai"] if args.openai else []):
        emb, loja, clf = construir(nome, dataset, salvar=False)
        llm = LLM() if nome == "openai" else LLM(_ClienteSemRede())
        svc = VozService(emb, loja, clf, llm, Transcritor(), config.limiares(nome))
        aceit.append(avaliar_aceitacao(nome + (" + GPT juiz" if nome == "openai" else " (fallback, sem LLM)"), svc))

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metricas.json").write_text(
        json.dumps({"duplicadas": dup, "categoria": cat, "falsos_alarmes": erros, "aceitacao": aceit}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    figuras(curvas)
    escrever_relatorio(dup, cat, erros, args.openai, aceit)
    print(json.dumps({"duplicadas": dup, "categoria": [{k: v for k, v in c.items() if k not in ("erros",)} for c in cat]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
