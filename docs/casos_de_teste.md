# Casos de teste e critérios de avaliação

Todos os testes `pytest` rodam **sem rede e sem chave** (cliente OpenAI simulado em `tests/conftest.py`). As métricas com dados estão em [`reports/avaliacao.md`](../reports/avaliacao.md).

## Critérios
| Tarefa | Métrica | Como é medida |
|---|---|---|
| Duplicadas (recuperação) | AUC-ROC, PR-AUC, precisão/recall/F1, Hit@1/3, MRR | pares do conjunto TESTE (limiar calibrado no DEV) |
| Duplicadas (uso real) | detectada x falso alarme | consulta contra o índice; "nova" remove todas as variantes do problema |
| Agrupamento | ARI, homogeneidade, completude | ligação média, só TESTE |
| Aceitação ponta a ponta | acurácia, recall de duplicadas, falso alarme | 40 propostas inéditas (`data/aceitacao.jsonl`) no `VozService.analisar` |
| Categoria | acurácia, F1 macro vs. baseline majoritário | GroupKFold por problema |
| Transcrição | funcional e de falha (WER **não medido**: sem conjunto de áudios de referência; `scripts/avaliar_audio.py` permite medir) | `test_audio_ok_e_invalidos`, `test_transcricao_*` |

## Casos positivos (comportamento esperado)
| Caso | Teste |
|---|---|
| Paráfrase recupera a proposta original no mesmo bairro | `test_parafrase_recupera_o_original_no_mesmo_bairro` |
| Fluxo completo: organiza, categoriza, avisa duplicada | `test_fluxo_completo_com_llm` |
| Proposta nova não gera alerta | `test_proposta_nova_nao_gera_alerta` |
| Indexar e depois detectar | `test_indexar_novo_item_e_detectar_depois` |
| Agrupar variantes do mesmo problema | `test_agrupar_junta_variantes` |
| Áudio é transcrito e analisado | `test_audio_ok_e_invalidos` |
| Resumo neutro e rascunho de parecer | `test_resumo_e_parecer_com_fallback` |

## Casos negativos / falhas
| Caso | Teste |
|---|---|
| Busca não vaza para outro bairro | `test_nao_recupera_de_outro_bairro` |
| LLM rebaixa falso alarme (mesma rua, outro problema) | `test_juiz_rebaixa_falso_alarme_para_parecida` |
| LLM fora do ar / sem chave / resposta inválida | `test_falha_de_rede_vira_indisponivel`, `test_sem_chave_vira_indisponivel`, `test_resposta_invalida_vira_indisponivel`, `test_sem_llm_usa_fallback_e_continua` |
| LLM inventa id, número ou prazo | `test_mensagem_duplicada_valida_ids_e_numeros`, `test_resumo_rejeita_numero_inventado`, `test_parecer_rejeita_prazo_inventado` |
| Fallback não afirma "já existe" se só é parecida | `test_fallback_nao_afirma_igualdade_quando_so_relacionada` |
| Embeddings fora do ar não bloqueiam | `test_falha_no_embedder_nao_bloqueia` |
| Dados pessoais não chegam ao LLM nem ao dataset | `test_pii_nao_chega_ao_llm`, `test_mascara_cpf_telefone_email`, `test_dataset_sem_dados_pessoais` |
| Prompt injection (texto do cidadão tratado como dado) | `test_organizar_ok_e_chamada_segura` |
| Áudio inválido, vazio, grande, sem fala; serviço fora | `test_transcricao_valida_entrada`, `test_transcricao_ok_vazia_e_falha`, `test_audio_servico_fora_do_ar_503` |
| Chave errada / entrada inválida | `test_autenticacao`, `test_entrada_invalida_422`, `test_parecer_decisao_invalida_422` |
| WER | `test_wer_*` |

## Falhas conhecidas (medidas, não escondidas)
- Baseline lexical (TF-IDF, sem LLM): muitos falsos alarmes em propostas novas do mesmo bairro; por isso existe a etapa 2 (GPT confirma "mesmo problema") e o aviso cauteloso ("parecidas") quando não há confirmação.
- Categoria por TF-IDF é fraca (semântica); o serviço só sugere e marca `precisa_revisao` abaixo de 0,50.
- Nomes e endereços não são mascarados (limite documentado em `docs/arquitetura.md`).
