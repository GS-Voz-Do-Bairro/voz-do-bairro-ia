"""Gera data/aceitacao.jsonl: 40 propostas INÉDITAS (fora do dataset de treino) para o teste de aceitação.

- 20 'duplicada': redação nova de um problema que já está no índice (esperado: decisão ja_existe com o id certo);
- 10 'nova_facil': assunto que não existe no bairro;
- 10 'nova_dificil': mesma rua/equipamento de uma proposta existente, mas problema DIFERENTE (esperado: não é duplicada).
"""
import json
from pathlib import Path

JA, VN, CE, PI = "Jardim das Acácias", "Vila Nova", "Centro", "Parque Industrial"

DUP = [
    ("m01", JA, "Moro perto da padaria e a rua das flores virou um buraco só, ontem um carro ficou preso nele"),
    ("m02", JA, "o farol da avenida brasil com a rua das flores parou de funcionar, cuidado quem passa"),
    ("m03", VN, "falta pintar as listras pra atravessar na frente da escola santos dumont"),
    ("m04", CE, "Os idosos que esperam o ônibus na XV de Novembro ficam expostos à chuva, falta um abrigo."),
    ("m05", PI, "Quem trabalha nas fábricas precisa de mais viagens do 305 de manhã cedo"),
    ("m06", JA, "Impossível andar de cadeira de rodas na calçada da Rua das Flores, está esburacada"),
    ("m07", VN, "Ciclistas da Avenida das Palmeiras precisam de espaço próprio, uma faixa só pra bicicleta"),
    ("z01", JA, "a rua das flores está um breu à noite porque as lâmpadas dos postes queimaram"),
    ("z02", VN, "Terreno abandonado da Santos Dumont acumulando lixo e bicho, precisa limpar"),
    ("z03", CE, "O lixo do Centro não está sendo recolhido, já faz dias que nenhum caminhão passa"),
    ("z04", PI, "Quando chove a Rua das Indústrias alaga porque o bueiro está entupido de lixo"),
    ("z05", JA, "Os galhos da árvore da Rua dos Ipês estão encostando na rede elétrica, pode cortar?"),
    ("z08", JA, "O esgoto está escorrendo na Rua dos Ipês, tem mau cheiro e mosquito"),
    ("s01", VN, "A UBS da Vila Nova está sem médico faz um tempão, só tem atendimento de enfermagem"),
    ("s02", CE, "No posto do Centro acabou o remédio da pressão, meu pai ficou sem"),
    ("s03", JA, "Esperei meses por uma consulta no posto das Acácias, precisa de mais vagas"),
    ("s04", PI, "Pneus velhos com água acumulada no Parque Industrial podem causar dengue"),
    ("s07", PI, "O SAMU leva muito tempo pra chegar no Parque Industrial, precisa de base mais perto"),
    ("l01", JA, "O balanço e o escorregador da praça das Acácias estão quebrados, criança pode se ferir"),
    ("l06", CE, "Queremos a biblioteca do Centro aberta de novo, ela está fechada há muito tempo"),
]
NOVA_FACIL = [
    (JA, "Instalar um bicicletário seguro no terminal de ônibus do Jardim das Acácias"),
    (JA, "Criar uma horta comunitária no bairro para as famílias plantarem"),
    (VN, "Instalar um posto de vacinação contra a gripe no mercado da Vila Nova"),
    (VN, "Colocar lixeiras de coleta seletiva nas ruas da Vila Nova"),
    (CE, "Criar uma área de embarque para carros de aplicativo no Centro"),
    (CE, "Promover shows gratuitos de música na Praça Central nos fins de semana"),
    (CE, "Programa de caminhadas orientadas para a saúde dos idosos do Centro"),
    (PI, "Pavimentar a rua de terra que dá acesso aos galpões do Parque Industrial"),
    (PI, "Oferecer exames de vista gratuitos para os trabalhadores do Parque Industrial"),
    (PI, "Construir uma quadra de vôlei de areia no Parque Industrial"),
]
NOVA_DIFICIL = [
    (JA, "Falta lixeira na Rua das Flores, o povo joga lixo no chão"),
    (JA, "Pedimos uma lombada na Rua das Flores porque os carros passam muito rápido"),
    (JA, "A Praça das Acácias precisa de mais bancos para os idosos sentarem"),
    (JA, "A sala de espera da UBS do Jardim das Acácias não tem ventilador nem cadeiras suficientes"),
    (VN, "Falta bebedouro na quadra esportiva da Vila Nova"),
    (VN, "O posto de saúde da Vila Nova precisa de rampa de acessibilidade na entrada"),
    (CE, "Instalar paraciclo na Rua XV de Novembro"),
    (CE, "A UBS do Centro precisa de uma sala de vacinação"),
    (PI, "A Rua das Indústrias precisa de calçada, os trabalhadores andam no meio da rua"),
    (PI, "Instalar iluminação na pista de caminhada do Parque Industrial"),
]


def main():
    saida = Path(__file__).resolve().parent.parent / "data" / "aceitacao.jsonl"
    linhas = [{"tipo": "duplicada", "issue_esperado": i, "bairro": b, "texto": t} for i, b, t in DUP]
    linhas += [{"tipo": "nova_facil", "issue_esperado": None, "bairro": b, "texto": t} for b, t in NOVA_FACIL]
    linhas += [{"tipo": "nova_dificil", "issue_esperado": None, "bairro": b, "texto": t} for b, t in NOVA_DIFICIL]
    saida.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in linhas) + "\n", encoding="utf-8")
    print(len(linhas), "casos ->", saida)


if __name__ == "__main__":
    main()
