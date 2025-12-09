# RAG and Search

Build intelligent retrieval systems with open-source models. Learn semantic search, reranking, and retrieval-augmented generation patterns.

## 📚 Notebooks

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [Text RAG](Text_RAG.ipynb) | Basic RAG implementation with vector embeddings and similarity search | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Text_RAG.ipynb) |
| [Semantic Search](Semantic_Search.ipynb) | Implement vector search with embedding models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Semantic_Search.ipynb) |
| [Search with Reranking](Search_with_Reranking.ipynb) | Improve search results using reranker models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Search_with_Reranking.ipynb) |
| [Open Contextual RAG](Open_Contextual_RAG.ipynb) | Implementation of Contextual Retrieval using open models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Open_Contextual_RAG.ipynb) |
| [RAG with Reasoning Models](RAG_with_Reasoning_Models.ipynb) | RAG + source citations with DeepSeek R1 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/RAG_with_Reasoning_Models.ipynb) |
| [Embedding Visualization](Embedding_Visualization.ipynb) | Visualize vector embeddings with UMAP clustering | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Embedding_Visualization.ipynb) |

## 🚀 Production Applications

| Application | Description |
| ----------- | ----------- |
| [Contextual RAG on Union](contextual_rag_on_union/) | Full-stack RAG with FastAPI + Gradio, featuring hybrid search (BM25 + vector), reranking, and Milvus |

## 🎯 Key Concepts

- **Vector Embeddings**: Converting text to numerical representations for semantic search
- **Similarity Search**: Finding relevant documents using cosine similarity or other metrics
- **Reranking**: Re-scoring retrieved results for better relevance
- **Hybrid Search**: Combining keyword (BM25) and semantic (vector) search
- **Contextual Retrieval**: Augmenting chunks with context for improved retrieval

## 📊 Common Patterns

1. **Basic RAG Pipeline**: Embed → Store → Retrieve → Generate
2. **Hybrid Retrieval**: BM25 + Vector Search → Reciprocal Rank Fusion
3. **Advanced RAG**: Contextual Chunks → Hybrid Search → Reranking → Reasoning LLM

## Prerequisites

- Together AI API key
- Understanding of vector databases (Milvus, ChromaDB, etc.)
- Basic knowledge of embeddings and similarity metrics
