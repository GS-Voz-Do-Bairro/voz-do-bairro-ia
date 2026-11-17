# Guia de adaptação: onde mexer quando o requisito muda

| Mudança pedida | Onde mexer | Esforço |
|---|---|---|
| Nova categoria (ex.: segurança, educação) | `config.CATEGORIAS`, incluir exemplos em `scripts/gerar_dataset.py`, rodar `python -m vozia.evaluate` | baixo |
| Trocar de provedor de LLM (Azure OpenAI, Gemini, modelo local via Ollama) | `vozia/llm.py` (só `_chamar` e `client`) e `vozia/embeddings.py` (classe com `embed`); o resto não muda | baixo |
| Rodar 100% local, sem enviar dado à nuvem | `VOZIA_EMBEDDER=tfidf` já funciona; trocar embeddings por `sentence-transformers` e o LLM por Ollama implementando a mesma interface | médio |
| Deixar a detecção de duplicadas mais rígida ou mais branda | limiares em `models/<embedder>/limiares.json` (calibrados por `evaluate`) | baixo |
| Duplicadas entre bairros vizinhos | `VectorStore.buscar` (parâmetro `bairro`) e `similar.buscar_similares` | baixo |
| Moderação de conteúdo ofensivo antes de publicar | novo passo em `servico.analisar` (classificador ou endpoint de moderação) | médio |
| Mais idiomas | prompts em `llm.py` e `language` em `transcribe.py` | baixo |
| Muitas propostas (milhões) | trocar `VectorStore` por pgvector/Qdrant mantendo a interface `upsert/buscar/remover` | médio |
| Resumos mais longos ou outro tom | instruções em `llm.py` e limites em `Resumo`/`Parecer` | baixo |
| Auditar o que o LLM respondeu | registrar `modo`, `verificacao` e id da proposta (já presentes na resposta) em log estruturado | baixo |
