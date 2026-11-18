# Evaluation report: `run-20260930T231441-ff8ffc4b`

## Retrieval + answer-quality metrics, per (dataset, strategy)

| dataset | strategy | recall_at_k | ndcg_at_k | mrr_at_k | map_at_k | mean_answer_score | num_generated | num_queries |
|---|---|---|---|---|---|---|---|---|
| fiqa | agentic_corrective | 0.3326 | 0.2782 | 0.3482 | 0.2183 | 0.0000 | 15 | 23 |
| fiqa | dense | 0.4340 | 0.3640 | 0.4416 | 0.2885 | 0.0000 | 15 | 648 |
| fiqa | hybrid_reranked | 0.3326 | 0.2782 | 0.3482 | 0.2183 | 0.0000 | 15 | 23 |
| fiqa | hybrid_rrf | 0.4057 | 0.3333 | 0.4098 | 0.2577 | 0.0000 | 15 | 648 |
| fiqa | sparse_bm25 | 0.2527 | 0.2058 | 0.2642 | 0.1544 | 0.0000 | 15 | 648 |
| multihop_rag | agentic_corrective | 0.3821 | 0.3801 | 0.5557 | 0.2924 | 0.0238 | 35 | 53 |
| multihop_rag | dense | 0.4722 | 0.4394 | 0.5724 | 0.3490 | 0.0340 | 35 | 1500 |
| multihop_rag | hybrid_reranked | 0.3821 | 0.3801 | 0.5557 | 0.2924 | 0.0200 | 35 | 53 |
| multihop_rag | hybrid_rrf | 0.5610 | 0.5256 | 0.6530 | 0.4348 | 0.0197 | 35 | 1500 |
| multihop_rag | sparse_bm25 | 0.5648 | 0.5352 | 0.6580 | 0.4504 | 0.0254 | 35 | 1500 |
| nfcorpus | agentic_corrective | 0.0720 | 0.2273 | 0.3636 | 0.1984 | 0.0000 | 8 | 11 |
| nfcorpus | dense | 0.1126 | 0.2346 | 0.4297 | 0.1577 | 0.0000 | 8 | 323 |
| nfcorpus | hybrid_reranked | 0.0720 | 0.2273 | 0.3636 | 0.1984 | 0.0000 | 8 | 11 |
| nfcorpus | hybrid_rrf | 0.1282 | 0.2564 | 0.4670 | 0.1721 | 0.0000 | 8 | 323 |
| nfcorpus | sparse_bm25 | 0.1175 | 0.2311 | 0.4197 | 0.1532 | 0.0000 | 8 | 323 |
| scifact | agentic_corrective | 0.8182 | 0.6875 | 0.6465 | 0.6465 | 0.0000 | 7 | 11 |
| scifact | dense | 0.7453 | 0.6147 | 0.5811 | 0.5672 | 0.0000 | 7 | 300 |
| scifact | hybrid_reranked | 0.8182 | 0.6875 | 0.6465 | 0.6465 | 0.0000 | 7 | 11 |
| scifact | hybrid_rrf | 0.7892 | 0.6649 | 0.6314 | 0.6205 | 0.0000 | 7 | 300 |
| scifact | sparse_bm25 | 0.7312 | 0.6226 | 0.5938 | 0.5826 | 0.0000 | 7 | 300 |
| t2_ragbench | agentic_corrective | 0.5472 | 0.3350 | 0.2687 | 0.2687 | 0.0000 | 35 | 53 |
| t2_ragbench | dense | 0.3107 | 0.2020 | 0.1680 | 0.1680 | 0.0000 | 35 | 1500 |
| t2_ragbench | hybrid_reranked | 0.5472 | 0.3350 | 0.2687 | 0.2687 | 0.0000 | 35 | 53 |
| t2_ragbench | hybrid_rrf | 0.6660 | 0.4048 | 0.3244 | 0.3244 | 0.0000 | 35 | 1500 |
| t2_ragbench | sparse_bm25 | 0.7547 | 0.5352 | 0.4641 | 0.4641 | 0.0571 | 35 | 1500 |


## Agentic pipeline vs. each baseline (paired bootstrap, answer_score)

| dataset | baseline | n | mean_difference | ci_low | ci_high | p_value |
|---|---|---|---|---|---|---|
| fiqa | sparse_bm25 | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| fiqa | dense | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| fiqa | hybrid_rrf | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| fiqa | hybrid_reranked | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| multihop_rag | sparse_bm25 | 35 | -0.0016 | -0.0459 | 0.0431 | 0.9464 |
| multihop_rag | dense | 35 | -0.0101 | -0.0610 | 0.0378 | 0.7504 |
| multihop_rag | hybrid_rrf | 35 | 0.0041 | -0.0270 | 0.0378 | 0.8188 |
| multihop_rag | hybrid_reranked | 35 | 0.0038 | 0.0000 | 0.0114 | 0.7252 |
| nfcorpus | sparse_bm25 | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| nfcorpus | dense | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| nfcorpus | hybrid_rrf | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| nfcorpus | hybrid_reranked | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| scifact | sparse_bm25 | 7 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| scifact | dense | 7 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| scifact | hybrid_rrf | 7 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| scifact | hybrid_reranked | 7 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| t2_ragbench | sparse_bm25 | 35 | -0.0571 | -0.1429 | 0.0000 | 0.2510 |
| t2_ragbench | dense | 35 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| t2_ragbench | hybrid_rrf | 35 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| t2_ragbench | hybrid_reranked | 35 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |


## Cost & latency, per (dataset, strategy)

| dataset | strategy | mean_retrieval_latency_s | mean_generation_latency_s | total_prompt_tokens | total_completion_tokens | estimated_cost_usd |
|---|---|---|---|---|---|---|
| fiqa | agentic_corrective | 4.3078 | 0.9830 | 7680 | 144 | 0.0000 |
| fiqa | dense | 0.0289 | 0.9496 | 7606 | 131 | 0.0000 |
| fiqa | hybrid_reranked | 4.6229 | 1.1395 | 7680 | 144 | 0.0000 |
| fiqa | hybrid_rrf | 0.5349 | 1.0161 | 7680 | 152 | 0.0000 |
| fiqa | sparse_bm25 | 0.5134 | 1.2021 | 7680 | 216 | 0.0000 |
| multihop_rag | agentic_corrective | 9.0195 | 1.6569 | 17408 | 755 | 0.0000 |
| multihop_rag | dense | 0.0325 | 1.5531 | 17920 | 765 | 0.0000 |
| multihop_rag | hybrid_reranked | 8.0919 | 1.5560 | 17920 | 760 | 0.0000 |
| multihop_rag | hybrid_rrf | 2.0475 | 1.5133 | 17920 | 747 | 0.0000 |
| multihop_rag | sparse_bm25 | 2.4020 | 1.7102 | 17920 | 917 | 0.0000 |
| nfcorpus | agentic_corrective | 4.9932 | 1.2017 | 4096 | 108 | 0.0000 |
| nfcorpus | dense | 0.0275 | 0.8249 | 4096 | 48 | 0.0000 |
| nfcorpus | hybrid_reranked | 5.1535 | 1.1707 | 4096 | 108 | 0.0000 |
| nfcorpus | hybrid_rrf | 0.1792 | 0.9948 | 4096 | 69 | 0.0000 |
| nfcorpus | sparse_bm25 | 0.1616 | 1.0243 | 4096 | 81 | 0.0000 |
| scifact | agentic_corrective | 5.9040 | 1.2705 | 3584 | 105 | 0.0000 |
| scifact | dense | 0.0390 | 1.4485 | 3584 | 139 | 0.0000 |
| scifact | hybrid_reranked | 6.3250 | 1.2325 | 3584 | 105 | 0.0000 |
| scifact | hybrid_rrf | 0.5825 | 1.1357 | 3584 | 87 | 0.0000 |
| scifact | sparse_bm25 | 0.6430 | 1.1969 | 3584 | 71 | 0.0000 |
| t2_ragbench | agentic_corrective | 8.6118 | 3.4120 | 17920 | 2165 | 0.0000 |
| t2_ragbench | dense | 0.0297 | 3.2196 | 17920 | 1955 | 0.0000 |
| t2_ragbench | hybrid_reranked | 8.6132 | 3.4449 | 17920 | 2165 | 0.0000 |
| t2_ragbench | hybrid_rrf | 1.1389 | 4.2089 | 17920 | 2794 | 0.0000 |
| t2_ragbench | sparse_bm25 | 1.0972 | 3.7488 | 17920 | 2445 | 0.0000 |
