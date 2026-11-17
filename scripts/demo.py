"""Demonstração rápida do fluxo (usa o serviço rodando em VOZIA_URL, padrão http://localhost:8000).

    python scripts/demo.py
"""
import json
import os
import urllib.request

URL = os.getenv("VOZIA_URL", "http://localhost:8000").rstrip("/")
CHAVE = os.getenv("AI_SERVICE_API_KEY", "")

CASOS = [
    ("Parafraseia uma proposta que já existe", "o buraco da rua das flores perto da padaria já derrubou várias motos", "Jardim das Acácias"),
    ("Outro problema existente na mesma rua (lâmpadas já têm proposta)", "as lâmpadas dos postes da rua das flores queimaram", "Jardim das Acácias"),
    ("Problema NOVO na mesma rua (não deve virar ja_existe)", "pedimos uma lombada na rua das flores porque os carros passam muito rápido", "Jardim das Acácias"),
    ("Proposta nova", "queremos aulas gratuitas de xadrez para aposentados na praça", "Jardim das Acácias"),
    ("Com dado pessoal (deve ser mascarado)", "buraco na rua das flores, falar com Maria no (11) 91234-5678", "Jardim das Acácias"),
]


def post(caminho, corpo):
    req = urllib.request.Request(URL + caminho, data=json.dumps(corpo).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": CHAVE})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


if __name__ == "__main__":
    for titulo, texto, bairro in CASOS:
        r = post("/propostas/analisar", {"texto": texto, "bairro": bairro})
        print(f"\n== {titulo}\n   texto: {texto}")
        print(f"   decisão: {r['decisao']} (verificação: {r['verificacao']}, modo: {r['modo']})")
        print(f"   categoria: {r['categoria']['categoria']} ({r['categoria']['confianca']})  PII mascarados: {r['pii_mascarados']}")
        if r["mensagem_ao_cidadao"]:
            print(f"   mensagem: {r['mensagem_ao_cidadao']['mensagem']}")
