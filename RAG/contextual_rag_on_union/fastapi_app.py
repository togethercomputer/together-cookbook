import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Generator, Optional

import bm25s
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymilvus import MilvusClient
from together import Together

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "Salesforce/Llama-Rank-V1"
FINAL_RESPONSE_MODEL = "deepseek-ai/DeepSeek-R1"
data = {}


def get_embedding(chunk: str, model_api_string: str):
    outputs = client.embeddings.create(
        input=chunk,
        model=model_api_string,
    )
    return outputs.data[0].embedding


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(os.getenv("CONTEXTUAL_CHUNKS_JSON"), "r", encoding="utf-8") as json_file:
        contextual_chunks_data = json.load(json_file)

    data["contextual_chunks_data"] = contextual_chunks_data

    client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
    data["client"] = client

    retriever = bm25s.BM25()
    bm25_index = retriever.load(save_dir=os.getenv("BM25S_INDEX"))
    data["bm25_index"] = bm25_index

    yield


def vector_retrieval(
    query: str,
    model_api_string: str,
    top_k: int = 5,
) -> list[str]:
    query_embeddings = get_embedding(query, model_api_string)
    query_embeddings_np = np.array(query_embeddings, dtype=np.float32)

    result = data["client"].search(
        data["client"].list_collections()[0],
        [query_embeddings_np],
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        anns_field="embedding",
        output_fields=["document_index", "title"],
    )

    ids = []
    for hits in result:
        for hit in hits:
            ids.append(hit["entity"]["document_index"])

    return ids


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
    query: str, retrieved_chunks: str, history: Optional[list[dict[str, str]]] = None
) -> Generator[str, None, None]:
    if history:
        history[-1][
            "content"
        ] = f"Answer the question: {query}. Here is relevant information: {retrieved_chunks}"

    response = client.chat.completions.create(
        messages=history
        or [
            {
                "role": "user",
                "content": f"Answer the question: {query}. Here is relevant information: {retrieved_chunks}",
            }
        ],
        model=FINAL_RESPONSE_MODEL,
        stream=True,
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    history: Optional[list] = None


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    query = request.query
    history = request.history or []

    vector_top_k = vector_retrieval(
        query=query, model_api_string="BAAI/bge-large-en-v1.5"
    )
    bm25_top_k = bm25_retrieval(query=query)

    hybrid_top_k_ids = reciprocal_rank_fusion(vector_top_k, bm25_top_k)
    hybrid_top_k_docs = [
        data["contextual_chunks_data"][index] for index in hybrid_top_k_ids
    ]

    retrieved_chunks = rerank(query=query, hybrid_top_k_docs=hybrid_top_k_docs)

    return StreamingResponse(
        generate_final_response(query, retrieved_chunks, history),
        media_type="text/plain",
    )
