import os

import gradio as gr
import requests


def stream_query_to_fastapi(query, history):
    history.append({"role": "user", "content": query})

    try:
        response = requests.post(
            os.environ["FASTAPI_ENDPOINT"] + "/chat",
            json={"query": query, "history": history},
            stream=True,
        )
        response.raise_for_status()

        result = ""
        for chunk in response.iter_content(chunk_size=128):
            result += chunk.decode("utf-8")
            yield result
    except Exception as e:
        yield f"An error occurred: {str(e)}"


demo = gr.ChatInterface(
    stream_query_to_fastapi,
    type="messages",
    title="Paul Graham Insights Hub",
    description="Enter a query, and this app will search through Paul Graham's essays to find the most relevant passages that answer your question.",
)

demo.launch(server_port=8080)
