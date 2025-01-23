import os

import gradio as gr
import requests


def stream_query_to_fastapi(query: str):
    """
    Sends the query to the FastAPI endpoint and streams the response.
    """
    try:
        response = requests.post(
            os.environ["FASTAPI_ENDPOINT"] + "/chat",
            params={"query": query},
            stream=True,
        )
        response.raise_for_status()

        result = ""
        for chunk in response.iter_content(chunk_size=128):
            result += chunk.decode("utf-8")
            yield result
    except Exception as e:
        yield f"An error occurred: {str(e)}"


demo = gr.Interface(
    fn=stream_query_to_fastapi,
    inputs=gr.Textbox(label="Enter your query:"),
    outputs=gr.Textbox(label="Answer:"),
    title="Query Paul Graham's Essays",
    description="Enter a query, and this app will search through Paul Graham's essays to find the most relevant passages that answer your question.",
    flagging_mode="never",
)

demo.launch(server_port=8080)
