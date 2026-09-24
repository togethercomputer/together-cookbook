# Archived notebooks

Notebooks in this folder are no longer maintained and are excluded from execution CI. They were archived because they depend on models that are no longer available on the Together platform, or because they have been superseded. They're kept for reference; the code may not run as written.

| Notebook | Archived | Why |
| -------- | -------- | --- |
| [Text RAG](Text_RAG.ipynb) | 2026-09 | Embeds with `intfloat/multilingual-e5-large-instruct`, removed from serverless 2026-09 with no replacement (serverless no longer offers embedding models). |
| [Semantic Search](Semantic_Search.ipynb) | 2026-09 | Same removed embeddings model; no serverless embedding models remain. |
| [Search with Reranking](Search_with_Reranking.ipynb) | 2026-09 | Same removed embeddings model, and serverless offers no rerank models (rerank is dedicated-endpoints only). |
| [Open Contextual RAG](Open_Contextual_RAG.ipynb) | 2026-09 | Same removed embeddings model. |
| [Contextual RAG on Union](contextual_rag_on_union/Contextual_RAG_on_Union.ipynb) | 2026-09 | Same removed embeddings model (Union.ai deployment variant of Open Contextual RAG). |
| [RAG with Reasoning Models](RAG_with_Reasoning_Models.ipynb) | 2026-09 | Same removed embeddings model, plus DeepSeek-R1, which is no longer on serverless. |
| [Embedding Visualization](Embedding_Visualization.ipynb) | 2026-09 | Same removed embeddings model; no serverless embedding models remain. |
| [Flux LoRA Inference](Flux_LoRA_Inference.ipynb) | 2026-09 | `black-forest-labs/FLUX.1-krea-dev` (the only serverless image-LoRA serving) was deprecated 2026-05-27 with no replacement; `image_loras` is rejected by the current FLUX.2 models. |
| [LoRA Inference and Fine-tuning](LoRA_Finetuning%26Inference.ipynb) | 2026-09 | Serverless LoRA inference (the `-adapter` serving path this notebook demonstrates) has been discontinued; fine-tuned adapters now serve on dedicated endpoints. |
| [Together Code Interpreter](Together_Code_Interpreter.ipynb) | 2026-09 | Together Code Interpreter (TCI) has been removed from the platform; the `/v1/tci/execute` endpoint no longer exists. |
| [Together Open Data Science Agent](DataScienceAgent/Together_Open_DataScience_Agent.ipynb) | 2026-09 | Built on Together Code Interpreter, which has been removed from the platform. |
| [OpenEnv Code Interpreter](OpenEnv_Code_Interpreter/README.md) | 2026-09 | An OpenEnv environment wrapping Together Code Interpreter, which has been removed from the platform. |
| [Getting Started with Llama 4](Getting_started_with_Llama4.ipynb) | 2026-09 | Llama 4-era getting-started content, superseded by current model quickstarts in the [docs Guides tab](https://docs.together.ai/docs/guides). |
