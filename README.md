# Voz do Bairro IA

Funcionalidade de **inteligência artificial** da solução **CitizenTech / Voz do Bairro**: plataforma em que moradores propõem melhorias, a comunidade apoia e, ao atingir o mínimo de apoios, a proposta vira demanda oficial com prazo de resposta da prefeitura.

Esta IA ajuda a **incluir quem tem dificuldade com texto** e a **evitar propostas repetidas**:
- o morador pode **falar** a proposta: a IA transcreve e organiza;
- antes de publicar, o sistema avisa se **já existe proposta parecida** no bairro ("quer apoiar essa?"), para os apoios não se espalharem em 40 cópias;
- sugere a **categoria**; para o gestor, gera **resumo neutro** da discussão e **rascunho de parecer**.

**ODS:** 11 (cidades sustentáveis), 10 (redução das desigualdades) e 16 (instituições eficazes e transparentes).

> ⚠️ **Dados e resultados.** O dataset é **sintético** (escrito pela equipe, sem dados pessoais). As métricas medem a técnica e o pipeline, não o desempenho com moradores reais.

## Integrantes
| Nome | RM | Turma |
|---|---|---|
| Andrei de Paiva Gibbini | 563061 | 2TDSPF |
| Pedro Sakai Silva Zambaca | 565956 | 2TDSPF |
| Pedro Santos Pequini | 561842 | 2TDSPF |
| Arthur Câmara | 562310 | 2TDSPG |
| Diogo Cunha | 563654 | 2TDSPF |

## Funcionalidade e público
| Quem | O que ganha |
|---|---|
| **Cidadão com pouca escolaridade, idoso** | Envia a proposta por voz; a IA transcreve e escreve um título e uma descrição claros |
| **Qualquer cidadão** | Descobre proposta parecida antes de publicar e apoia em vez de duplicar; recebe sugestão de categoria |
| **Gestor da prefeitura** | Vê propostas agrupadas, lê um resumo neutro da discussão e recebe um rascunho de parecer (a decisão é dele) |
| **Comunidade** | Apoios concentrados em uma proposta, o que faz a meta ser atingida e a cobrança de prazo funcionar |

## Arquitetura
```
App mobile ──► API Java (Spring, JWT) ──► Serviço de IA (FastAPI) ──► OpenAI (GPT, Whisper, embeddings)
                     │                          │
                RabbitMQ / Oracle         índice vetorial por bairro + classificador de categoria
```
Fluxo de uma proposta: **áudio → Whisper → máscara de dados pessoais → GPT organiza → classificador sugere categoria → embeddings recuperam propostas parecidas do bairro → GPT confirma "é o mesmo problema?" → GPT escreve o aviso**. Diagrama completo e demais fluxos: [`docs/arquitetura.md`](docs/arquitetura.md). Integração com a API Java: [`docs/integracao_java.md`](docs/integracao_java.md).

## Modelos e tecnologias
| Tarefa | Modelo / técnica |
|---|---|
| Transcrição de áudio | OpenAI Whisper (`whisper-1`) |
| Organizar, mensagem, resumo, parecer, confirmação | OpenAI GPT (`gpt-4o-mini`, configurável) |
| Similaridade de propostas | Embeddings `text-embedding-3-small` + cosseno; baseline local TF-IDF de caracteres |
| Categoria | Regressão logística sobre os embeddings |
| Serviço | Python 3.12, FastAPI, Pydantic v2, scikit-learn, NumPy, Docker, pytest |

## Decisões técnicas e justificativas
- **RAG em duas etapas para duplicadas.** Embeddings recuperam candidatas rápido e barato; o LLM confirma se é *o mesmo problema*. Só com similaridade, propostas diferentes na mesma rua (buraco x iluminação) ficam parecidas demais e geram falso alarme, como a [avaliação](reports/avaliacao.md) mostra no baseline lexical.
- **Por que não treinar uma rede neural (Deep Learning)?** Com 144 textos, treinar uma rede do zero seria pior e mais caro que usar embeddings pré-treinados, que já são o resultado de Deep Learning. Usamos essa rede **pré-treinada** como extratora de características.
- **Por que OpenAI e não modelo local?** Qualidade em português, Whisper para áudio e custo de centavos neste volume. O trade-off é enviar texto à nuvem, mitigado por mascarar dados pessoais. A troca de provedor é simples: [`docs/guia_de_adaptacao.md`](docs/guia_de_adaptacao.md).
- **Por que um serviço Python separado?** O ecossistema de ML está em Python; a API Java só chama HTTP. Também mantém a chave da OpenAI fora do app.
- **O LLM não decide.** Ele escreve e confirma; ids e números citados são validados contra a fonte, e a publicação, o resumo e o parecer são sempre revisados por uma pessoa.

## Como executar
Pré-requisito: Python 3.11+.
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                 # preencha as chaves 
pytest                               # 50 testes, sem rede e sem chave
```
Subir o serviço:
```bash
export AI_SERVICE_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
export OPENAI_API_KEY=sk-...         
uvicorn app.main:app --port 8000     # Swagger em http://localhost:8000/docs
python scripts/demo.py               # roda 5 casos de demonstração
```
Windows (PowerShell): `$env:OPENAI_API_KEY="sk-..."`.

Exemplo:
```bash
curl -X POST localhost:8000/propostas/analisar -H "X-API-Key: $AI_SERVICE_API_KEY" -H "Content-Type: application/json" \
  -d '{"texto":"o buraco da rua das flores perto da padaria já derrubou várias motos","bairro":"Jardim das Acácias"}'
curl -X POST localhost:8000/audio/propostas -H "X-API-Key: $AI_SERVICE_API_KEY" -F "arquivo=@proposta.mp3" -F "bairro=Jardim das Acácias"
```
Docker: `docker build -t voz-ia . && docker run -p 8000:8000 -e AI_SERVICE_API_KEY=... -e OPENAI_API_KEY=... voz-ia`

Regerar dados e avaliação:
```bash
python scripts/gerar_dataset.py
python -m vozia.evaluate             # baselines locais, sem custo
python -m vozia.evaluate --openai    # inclui embeddings da OpenAI e a confirmação por GPT
```

## API
| Rota | Descrição |
|---|---|
| `GET /health` | Estado do serviço (sem chave) |
| `POST /propostas/analisar` | Organiza, sugere categoria e avisa se já existe proposta parecida |
| `POST /audio/propostas` | Igual, a partir de um áudio (multipart) |
| `POST /propostas/indexar` · `DELETE /propostas/{id}` | Mantém o índice das propostas publicadas |
| `POST /propostas/grupos` | Agrupa propostas parecidas do bairro |
| `POST /propostas/resumo` | Resumo neutro da proposta e dos comentários |
| `POST /pareceres/rascunho` | Rascunho de parecer para a decisão do gestor |

## Avaliação (resumo)
Detalhes, protocolo e falhas em [`reports/avaliacao.md`](reports/avaliacao.md). Teste = problemas que o sistema nunca viu; limiares calibrados só no DEV.

**Detector de duplicadas (conjunto de teste, executado):**

| Embedder | AUC-ROC | PR-AUC | Hit@1 | Hit@3 | MRR | Duplicada detectada | Falso alarme em proposta nova |
|---|---|---|---|---|---|---|---|
| palavras | 0.818 | 0.515 | 0.662 | 0.868 | 0.778 | 0.662 | 0.926 |
| tfidf | 0.904 | 0.769 | 0.912 | 0.985 | 0.947 | 0.897 | 0.912 |
| **openai** | **0.984** | **0.934** | **0.941** | **1.0** | **0.968** | **0.941** | 0.515 (só limiar) → **0.059** com confirmação do GPT |

**Classificador de categoria** (validação cruzada agrupada por problema): TF-IDF acerta **0.431**; embeddings da OpenAI acertam **0.812** (baseline da classe majoritária: 0.278).

**Outros resultados medidos (embeddings OpenAI + GPT):** agrupamento de propostas parecidas com ARI **0.963** (17 problemas reais, 18 grupos); confirmação por GPT com acurácia **0.979** por candidata; teste de aceitação ponta a ponta com 40 propostas inéditas: acurácia **0.95** (20/20 duplicadas detectadas, 10/10 novas fáceis sem alarme, 8/10 novas difíceis sem alarme). **Transcrição de áudio: não medida quantitativamente** (usamos o Whisper pré-treinado e não gravamos um conjunto de áudios de referência); está coberta por testes funcionais e de falha. O script `scripts/avaliar_audio.py` calcula o WER caso se queira medir depois.

**Interpretação.** Só com limiar de similaridade, mais da metade das propostas novas (51,5%) ainda disparava alarme falso, porque "mesma rua, outro problema" fica parecido. A confirmação pelo GPT derruba isso para **5,9%** mantendo 100% das duplicadas confirmadas: é a razão da arquitetura em duas etapas. No teste de aceitação, o que mais derruba os falsos alarmes é a confirmação do GPT: sem ela, o TF-IDF acerta 67,5%; com ela, TF-IDF e OpenAI ficam em 100% e 95% (diferença de 2 casos em 40, pequena demais para concluir que um embedder é melhor; os embeddings da OpenAI ganham com folga na recuperação e na categoria). Os dois erros restantes do teste de aceitação são veredito errado do GPT em propostas novas do Parque Industrial; por isso a decisão final de publicar é sempre do cidadão. O dataset é sintético e pequeno: os números medem a técnica, não o desempenho com moradores reais. Para reproduzir, rode `python -m vozia.evaluate --openai`.

**Testes (50, sem rede):** fluxo completo com LLM simulado, falhas do LLM e dos embeddings, resposta inválida, id ou número inventado, prompt injection, dados pessoais, áudio inválido, autenticação e limites de entrada.

**Detalhes completos** (agrupamento, aceitação ponta a ponta, falhas) em [`reports/avaliacao.md`](reports/avaliacao.md) e [`docs/casos_de_teste.md`](docs/casos_de_teste.md).

**Demo no navegador:** com o serviço rodando, abra `http://localhost:8000/demo` (texto ou gravação de voz).

## Uso responsável e limitações
- **Credenciais:** `OPENAI_API_KEY` e `AI_SERVICE_API_KEY` só por variável de ambiente; `.env` no `.gitignore`; só `.env.example` no repositório. A chave da OpenAI nunca vai ao app.
- **Dados pessoais (LGPD):** CPF, CNPJ, RG, telefone e e-mail são mascarados antes de qualquer envio ao LLM e antes de entrar no índice; o autor da proposta não é enviado à IA; comentários não são armazenados pelo serviço. **Limite:** nomes e endereços completos escritos no texto não são detectados.
- **Prompt injection:** o texto do cidadão vai como dado delimitado, com `<` e `>` removidos, e a saída é validada.
- **Limitações da IA:** pode errar a transcrição (sotaques, ruído), confundir problemas parecidos ou resumir mal; por isso o cidadão decide publicar, o gestor revisa resumo e parecer, e a categoria com baixa confiança vai para revisão. O dataset é pequeno e escrito pela equipe; a qualidade real só será conhecida com propostas reais. Há envio de texto à OpenAI (nuvem).
- **Falha segura:** sem LLM, o serviço continua com textos-modelo; sem embeddings, avisa e deixa publicar.

## Estrutura
```
vozia/      núcleo: config, privacy, embeddings, store, similar, categoria, llm, fallback, transcribe, servico, evaluate
app/        API FastAPI e contratos (schemas)
data/       propostas.jsonl, aceitacao.jsonl, audios/       models/   limiares calibrados (artefatos são gerados na 1ª execução)
tests/      50 testes                           reports/  avaliação e métricas
docs/       arquitetura, integração Java, guia de adaptação, figuras
scripts/    gerar_dataset/aceitacao, demo, avaliar_audio          Dockerfile, requirements*.txt, .env.example
```
