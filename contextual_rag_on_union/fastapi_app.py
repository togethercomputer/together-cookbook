import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Generator

import bm25s
import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from together import Together

client = Together(api_key="<YOUR_API_KEY>")  # TODO: App secrets


EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "Salesforce/Llama-Rank-V1"
FINAL_RESPONSE_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
data = {}


class TogetherEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = model_name

    def __call__(self, input: Documents) -> Embeddings:
        outputs = client.embeddings.create(
            input=input,
            model=self.model,
        )
        return [x.embedding for x in outputs.data]


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(os.getenv("CONTEXTUAL_CHUNKS_JSON"), "r", encoding="utf-8") as json_file:
        contextual_chunks_data = json.load(json_file)

    data["contextual_chunks_data"] = contextual_chunks_data

    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_DB_ENDPOINT"),
        port=8080,
    )
    collection = client.get_collection(
        client.list_collections()[0].name,
        embedding_function=TogetherEmbedding(model_name=EMBEDDING_MODEL),
    )

    data["collection"] = collection

    retriever = bm25s.BM25()
    bm25_index = retriever.load(save_dir=os.getenv("BM25S_INDEX"))

    data["bm25_index"] = bm25_index

    yield


def vector_retrieval(
    query: str,
    top_k: int = 5,
) -> list[str]:
    result = data["collection"].query(
        query_texts=[query],
        n_results=top_k,
    )

    return result["ids"][0]


def bm25_retrieval(query: str, k: int = 5) -> list[str]:
    results, _ = data["bm25_index"].retrieve(
        query_tokens=bm25s.tokenize(query),
        k=k,
        corpus=np.array(list(data["contextual_chunks_data"].values())),
    )

    # Create a mapping of values to keys for fast lookup
    value_to_key = {value: key for key, value in data["contextual_chunks_data"].items()}

    # Ensure the order of matching keys follows the same order as the results
    matching_keys = [value_to_key[doc] for doc in results[0]]

    return matching_keys


def reciprocal_rank_fusion(*list_of_list_ranks_system, K=60):
    # Dictionary to store RRF mapping
    rrf_map = defaultdict(float)

    # Calculate RRF score for each result in each list
    for rank_list in list_of_list_ranks_system:
        print(rank_list)
        for rank, item in enumerate(rank_list, 1):
            rrf_map[item] += 1 / (rank + K)

    # Sort items based on their RRF scores in descending order
    sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

    return [item for item, _ in sorted_items]


def rerank(query: str, hybrid_top_k_docs: list[str]) -> str:
    response = client.rerank.create(
        model=RERANKER_MODEL,
        query=query,
        documents=hybrid_top_k_docs,
        top_n=3,  # we only want the top 3 results but this can be a lot higher
    )

    retrieved_chunks = ""
    for result in response.results:
        retrieved_chunks += hybrid_top_k_docs[result.index] + "\n\n"

    return retrieved_chunks


def generate_final_response(
    query: str, retrieved_chunks: str
) -> Generator[str, None, None]:
    response = client.chat.completions.create(
        model=FINAL_RESPONSE_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {
                "role": "user",
                "content": f"Answer the question: {query}. Here is relevant information: {retrieved_chunks}",
            },
        ],
        stream=True,
    )

    # Stream the response in chunks
    for chunk in response:
        yield chunk.choices[0].delta.content


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(query: str) -> StreamingResponse:
    vector_top_k = vector_retrieval(query=query)
    bm25_top_k = bm25_retrieval(query=query)

    hybrid_top_k_ids = reciprocal_rank_fusion(vector_top_k, bm25_top_k)
    hybrid_top_k_docs = [
        data["contextual_chunks_data"][index] for index in hybrid_top_k_ids
    ]

    retrieved_chunks = rerank(query=query, hybrid_top_k_docs=hybrid_top_k_docs)

    return StreamingResponse(
        generate_final_response(query, retrieved_chunks), media_type="text/plain"
    )
