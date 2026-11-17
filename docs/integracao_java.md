# Integração com a API Java (Spring Boot)

O serviço de IA é um microsserviço HTTP. A API Java é a única que o chama; o app nunca fala direto com ele nem com a OpenAI. Assim a chave da OpenAI fica só no servidor.

## Variáveis no lado Java
| Variável | Para quê |
|---|---|
| `VOZIA_URL` | endereço do serviço de IA, ex. `http://ia:8000` |
| `AI_SERVICE_API_KEY` | a mesma chave do serviço, enviada em `X-API-Key` |

## Cliente (exemplo com `RestClient`)
```java
RestClient ia = RestClient.builder()
    .baseUrl(System.getenv("VOZIA_URL"))
    .defaultHeader("X-API-Key", System.getenv("AI_SERVICE_API_KEY"))
    .build();

record Analise(String texto, String bairro) {}

Map<?, ?> r = ia.post().uri("/propostas/analisar")
    .contentType(MediaType.APPLICATION_JSON)
    .body(new Analise(texto, bairro))
    .retrieve().body(Map.class);
```

## Onde cada rota entra
| Momento | Rota do serviço de IA |
|---|---|
| Cidadão digita ou grava a proposta (antes de publicar) | `POST /propostas/analisar` ou `POST /audio/propostas` (multipart) |
| Proposta publicada (consumidor do evento no RabbitMQ) | `POST /propostas/indexar` |
| Apoios mudam / proposta removida | `POST /propostas/indexar` (upsert) / `DELETE /propostas/{id}` |
| Gestor abre a demanda | `POST /propostas/resumo` |
| Gestor escolhe aceitar, recusar ou agendar | `POST /pareceres/rascunho` |
| Tela de gestão do bairro | `POST /propostas/grupos` |

## Erros
`401` chave inválida · `422` dado fora do limite · `400` áudio inválido · `503` transcrição indisponível. Se a IA cair, a API Java deve seguir sem as sugestões (a publicação não depende dela).
