"""Gera data/propostas.jsonl: propostas de moradores escritas à mão (sintéticas, sem dados pessoais).

Cada "issue" é um problema real do bairro; as variantes são formas diferentes de dizer a MESMA coisa
(formal, informal, com erros, estilo de áudio transcrito). Variantes da mesma issue são duplicadas entre si.
Há negativos difíceis de propósito (mesma rua ou mesmo equipamento, problemas diferentes).

Uso: python scripts/gerar_dataset.py
"""
import json
from pathlib import Path

JA, VN, CE, PI = "Jardim das Acácias", "Vila Nova", "Centro", "Parque Industrial"

ISSUES = [
    # ---------------- mobilidade ----------------
    ("m01", JA, "mobilidade", [
        "Tem um buraco enorme na Rua das Flores, em frente à padaria, e já quebrou roda de carro.",
        "Pedimos o conserto do buraco da Rua das Flores. Os carros desviam e quase batem.",
        "buraco na rua das flores perto da padaria ta perigoso demais, moto cai todo dia",
        "A Rua das Flores está com uma cratera no asfalto. Precisa tapar antes que aconteça acidente."]),
    ("m02", JA, "mobilidade", [
        "O semáforo do cruzamento da Avenida Brasil com a Rua das Flores está apagado há uma semana.",
        "Semáforo queimado na Av. Brasil. Ninguém respeita e já teve batida.",
        "o sinal da avenida brasil não funciona, as crianças da escola atravessam correndo",
        "Cruzamento sem semáforo funcionando na Avenida Brasil esquina com a Rua das Flores, urgente."]),
    ("m03", VN, "mobilidade", [
        "Falta faixa de pedestres na frente da Escola Municipal Santos Dumont.",
        "Pedimos pintura de faixa de pedestre na porta da escola Santos Dumont, os carros não param.",
        "as crianças atravessam sem faixa na escola santos dumont, precisa pintar",
        "Sem faixa de pedestre na entrada da escola, a rua Santos Dumont é movimentada."]),
    ("m04", CE, "mobilidade", [
        "O ponto de ônibus da Rua XV de Novembro não tem cobertura, idosos esperam no sol e na chuva.",
        "Precisamos de abrigo no ponto de ônibus da XV de Novembro.",
        "ponto de onibus sem cobertura na xv de novembro, quando chove todo mundo se molha",
        "Solicito a instalação de cobertura e banco no ponto de ônibus da Rua XV de Novembro."]),
    ("m05", PI, "mobilidade", [
        "A linha 305 passa só de hora em hora, os trabalhadores do Parque Industrial chegam atrasados.",
        "Aumentar a frequência do ônibus 305 no horário de pico.",
        "o 305 demora muito pra passar de manhã, precisa de mais carro",
        "Pedimos mais ônibus na linha 305 entre 6h e 8h para quem trabalha nas fábricas."]),
    ("m06", JA, "mobilidade", [
        "A calçada da Rua das Flores está toda quebrada, cadeirante não consegue passar.",
        "Calçada destruída na Rua das Flores, precisa de reforma e rampa de acessibilidade.",
        "minha mãe usa andador e não passa na calçada da rua das flores de tão quebrada",
        "Consertar o passeio da Rua das Flores e colocar rampa para cadeirantes."]),
    ("m07", VN, "mobilidade", [
        "Criar ciclovia na Avenida das Palmeiras, ciclistas disputam espaço com carros.",
        "Pedimos uma ciclovia na Av. das Palmeiras para segurança de quem pedala.",
        "ciclista corre risco na avenida das palmeiras, falta ciclofaixa",
        "Precisamos de ciclofaixa na Avenida das Palmeiras ligando a Vila Nova ao Centro."]),
    ("m08", VN, "mobilidade", [
        "Instalar lombada na Rua Santos Dumont, os carros passam em alta velocidade.",
        "Pedimos redutor de velocidade na Rua Santos Dumont, já teve atropelamento.",
        "os carro corre demais na santos dumont, precisa de lombada",
        "Solicito lombada ou radar na Rua Santos Dumont para reduzir a velocidade dos veículos."]),
    ("m09", CE, "mobilidade", [
        "Carros estacionam em cima da calçada da Rua XV de Novembro e bloqueiam os pedestres.",
        "Pedimos fiscalização de estacionamento irregular na calçada da XV de Novembro.",
        "tem carro parado na calçada da xv de novembro, a gente tem que andar na rua",
        "Os pedestres precisam descer para a rua porque veículos ocupam a calçada da Rua XV de Novembro."]),
    ("m10", PI, "mobilidade", [
        "Construir passarela sobre a rodovia para os trabalhadores do Parque Industrial atravessarem.",
        "Os funcionários das fábricas atravessam a rodovia correndo, precisamos de uma passarela.",
        "falta passarela na rodovia do parque industrial, o povo arrisca a vida pra atravessar",
        "Solicito uma passarela segura sobre a rodovia que divide o Parque Industrial."]),
    # ---------------- zeladoria ----------------
    ("z01", JA, "zeladoria", [
        "Os postes da Rua das Flores estão com a lâmpada queimada, a rua fica escura à noite.",
        "Falta iluminação na Rua das Flores, tem assalto no escuro.",
        "a luz da rua das flores apagou tudo, não dá pra andar a noite",
        "Solicito troca das lâmpadas queimadas dos postes da Rua das Flores."]),
    ("z02", VN, "zeladoria", [
        "Tem lixo e entulho acumulado no terreno baldio da Rua Santos Dumont, está cheio de ratos.",
        "Pedimos limpeza do terreno baldio da Santos Dumont, virou depósito de lixo.",
        "o terreno da rua santos dumont tá cheio de entulho e rato, cheiro horrível",
        "Entulho e lixo no terreno abandonado perto da escola, risco de ratos e doenças."]),
    ("z03", CE, "zeladoria", [
        "A coleta de lixo no Centro não passa há três dias, as sacolas se acumulam na calçada.",
        "Coleta de lixo atrasada no Centro, a rua está suja e fedendo.",
        "o caminhão do lixo não veio essa semana no centro",
        "Solicitamos regularizar a coleta de lixo no Centro, que está falhando."]),
    ("z04", PI, "zeladoria", [
        "O bueiro da Rua das Indústrias está entupido e alaga quando chove.",
        "Desentupir o bueiro da Rua das Indústrias, a água invade os galpões.",
        "bueiro entupido na rua das industrias, toda chuva alaga",
        "Pedimos a limpeza das galerias de água da Rua das Indústrias para evitar alagamento."]),
    ("z05", JA, "zeladoria", [
        "Uma árvore na Rua dos Ipês está com galhos sobre a fiação, precisa de poda.",
        "Poda urgente da árvore da Rua dos Ipês, os galhos tocam os fios.",
        "arvore da rua dos ipes quase caindo nos fios de luz",
        "Solicito poda de árvore na Rua dos Ipês antes da temporada de chuvas."]),
    ("z06", VN, "zeladoria", [
        "O mato está alto na calçada da Rua Santos Dumont e atrai bichos.",
        "Capinar o mato da calçada da Santos Dumont, está tomando conta.",
        "mato alto na rua santos dumont perto da escola, ninguém corta",
        "Pedimos roçada do mato na calçada da Rua Santos Dumont."]),
    ("z07", CE, "zeladoria", [
        "Os muros do Centro estão todos pichados, precisa de limpeza e pintura.",
        "Pichação em prédios e muros da Rua XV de Novembro, pedimos limpeza.",
        "pichação por todo o centro, ta feio demais",
        "Solicito limpeza e repintura dos muros pichados no Centro."]),
    ("z08", JA, "zeladoria", [
        "Tem esgoto a céu aberto na Rua dos Ipês, o cheiro é insuportável.",
        "Esgoto correndo na rua dos ipes, precisa de rede de esgoto.",
        "o esgoto da rua dos ipês corre pela rua e tem muito mosquito",
        "Pedimos a ligação da rede de esgoto na Rua dos Ipês, que hoje corre a céu aberto."]),
    ("z09", PI, "zeladoria", [
        "Caçambas de entulho abandonadas ocupam a calçada do Parque Industrial.",
        "Pedimos a retirada das caçambas de entulho paradas há meses no Parque Industrial.",
        "as caçamba de entulho tão paradas na calçada do parque industrial faz meses",
        "Remover caçambas abandonadas que bloqueiam a calçada no Parque Industrial."]),
    # ---------------- saúde ----------------
    ("s01", VN, "saúde", [
        "O posto de saúde da Vila Nova está sem clínico geral há dois meses.",
        "Falta médico no posto de saúde da Vila Nova, a fila é enorme.",
        "no posto da vila nova nao tem doutor faz meses, só enfermeira",
        "Solicitamos a contratação de um médico clínico geral para a UBS da Vila Nova."]),
    ("s02", CE, "saúde", [
        "Falta remédio de pressão na farmácia da UBS do Centro.",
        "A UBS do Centro está sem medicação para hipertensão há semanas.",
        "os remedios de pressão acabaram no posto do centro, idosos sem tratamento",
        "Pedimos o abastecimento de anti-hipertensivos na farmácia da UBS do Centro."]),
    ("s03", JA, "saúde", [
        "A marcação de consultas no posto do Jardim das Acácias demora mais de três meses.",
        "Demora enorme para agendar consulta na UBS do Jardim das Acácias.",
        "pra marcar consulta no posto das acacias é meses de espera",
        "Pedimos mais vagas de agendamento na UBS do Jardim das Acácias."]),
    ("s04", PI, "saúde", [
        "Há focos de dengue em pneus abandonados no Parque Industrial.",
        "Pedimos fiscalização de pneus e água parada contra a dengue no Parque Industrial.",
        "pneu velho com agua parada no parque industrial, vai dar dengue",
        "Combate à dengue: remover pneus e água parada nos galpões do Parque Industrial."]),
    ("s05", VN, "saúde", [
        "O posto de saúde da Vila Nova poderia abrir aos sábados para quem trabalha.",
        "Abrir a UBS da Vila Nova aos sábados de manhã.",
        "quem trabalha a semana toda não consegue ir ao posto, abrir sabado",
        "Pedimos atendimento aos sábados na UBS da Vila Nova."]),
    ("s06", CE, "saúde", [
        "Falta atendimento psicológico no posto do Centro.",
        "Precisamos de psicólogos na UBS do Centro, a fila é muito longa.",
        "nao tem psicologo no posto do centro pra atender",
        "Solicito atendimento de saúde mental na UBS do Centro."]),
    ("s07", PI, "saúde", [
        "A ambulância do SAMU demora quase uma hora para chegar ao Parque Industrial.",
        "Pedimos uma base do SAMU perto do Parque Industrial, o socorro demora demais.",
        "o samu demora uma hora pra chegar no parque industrial, já teve gente que passou mal",
        "Reduzir o tempo de resposta do SAMU no Parque Industrial."]),
    ("s08", JA, "saúde", [
        "Realizar campanha de vacinação nas escolas do Jardim das Acácias.",
        "Pedimos vacinação das crianças dentro das escolas do Jardim das Acácias.",
        "queria que vacinasse as criança na escola do jardim das acacias, os pais trabalham",
        "Solicito que a campanha de vacinação chegue às escolas do Jardim das Acácias."]),
    ("s09", VN, "saúde", [
        "Os exames de sangue demoram mais de um mês para sair na UBS da Vila Nova.",
        "Resultado de exame laboratorial demora demais na UBS da Vila Nova.",
        "fiz exame de sangue no posto da vila nova e até hoje nao saiu o resultado",
        "Pedimos agilidade na entrega dos exames laboratoriais da UBS da Vila Nova."]),
    # ---------------- lazer ----------------
    ("l01", JA, "lazer", [
        "Os brinquedos da Praça das Acácias estão quebrados e enferrujados, as crianças se machucam.",
        "Reformar o parquinho da Praça das Acácias, os brinquedos são perigosos.",
        "o parquinho da praça das acacias ta caindo aos pedaços",
        "Pedimos a troca dos brinquedos quebrados da Praça das Acácias."]),
    ("l02", VN, "lazer", [
        "A quadra esportiva da Vila Nova está sem rede, sem tabela e com o piso rachado.",
        "Reforma da quadra da Vila Nova: piso rachado e tabelas quebradas.",
        "quadra da vila nova nao tem mais tabela e o chão é todo rachado",
        "Solicito reforma da quadra de esportes da Vila Nova para os jovens."]),
    ("l03", CE, "lazer", [
        "Instalar academia ao ar livre na Praça Central para os idosos.",
        "Pedimos aparelhos de ginástica na Praça Central.",
        "queria uns aparelhos de exercício na praça central pros idosos",
        "Academia popular na Praça Central, com aparelhos para a terceira idade."]),
    ("l04", PI, "lazer", [
        "A pista de caminhada do Parque Industrial precisa de bancos e bebedouros.",
        "Colocar bancos e bebedouros na pista de caminhada do Parque Industrial.",
        "falta banco e agua pra beber na pista de caminhada do parque",
        "Pedimos bebedouros e bancos ao longo da pista de caminhada do Parque Industrial."]),
    ("l05", JA, "lazer", [
        "A Praça das Acácias é muito escura à noite, falta iluminação.",
        "Instalar refletores na Praça das Acácias, ninguém usa à noite por medo.",
        "a praça das acacias fica no escuro, tem medo de passar",
        "Pedimos mais postes de luz na Praça das Acácias."]),
    ("l06", CE, "lazer", [
        "Reabrir a biblioteca municipal do Centro, fechada há um ano.",
        "A biblioteca do Centro está fechada, queremos de volta.",
        "a biblioteca do centro fechou e nunca mais abriu",
        "Solicito a reabertura da biblioteca pública do Centro."]),
    ("l07", VN, "lazer", [
        "Iluminar o campo de futebol da Vila Nova para os jogos à noite.",
        "Pedimos refletores no campo de futebol da Vila Nova.",
        "o campo de futebol da vila nova nao tem luz, ninguém joga de noite",
        "Solicito iluminação para o campo de futebol de várzea da Vila Nova."]),
    ("l08", PI, "lazer", [
        "Criar uma feira cultural aos domingos no Parque Industrial.",
        "Pedimos eventos culturais e feira aos domingos no Parque Industrial.",
        "queria uma feira com música no domingo aqui no parque industrial",
        "Solicito programação cultural e uma feira de domingo para os moradores do Parque Industrial."]),
]


def main() -> None:
    saida = Path(__file__).resolve().parent.parent / "data" / "propostas.jsonl"
    saida.parent.mkdir(exist_ok=True)
    n = 0
    with saida.open("w", encoding="utf-8") as f:
        for issue_id, bairro, categoria, variantes in ISSUES:
            for k, texto in enumerate(variantes):
                n += 1
                f.write(json.dumps({
                    "id": f"{issue_id}-{k + 1}", "issue_id": issue_id, "bairro": bairro,
                    "categoria": categoria, "texto": texto,
                    "estilo": "coloquial" if texto[0].islower() else "formal",
                }, ensure_ascii=False) + "\n")
    print(f"{n} propostas, {len(ISSUES)} problemas -> {saida}")


if __name__ == "__main__":
    main()
