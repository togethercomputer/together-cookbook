# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "deepagents>=0.6,<0.8",
#     "langchain-together",
# ]
# ///
"""Content writer agent on Together AI's Nemotron 3 Ultra.

Ported from `deepagents/examples/deploy-content-writer` — the same brand-voice
memory (AGENTS.md) and skills (blog-post, social-media), but running locally on
`nvidia/nemotron-3-ultra-550b-a55b` via Together AI instead of a managed
LangSmith deployment. The per-user memory / auth layer of the original is
intentionally removed: every run is a fresh session and nothing persists
except the content files the agent writes to `output/`.

Usage:
    export TOGETHER_API_KEY=...
    uv run content_writer.py "Write a LinkedIn post about open-weight models"
"""

import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

HERE = Path(__file__).parent
MODEL = "together:nvidia/nemotron-3-ultra-550b-a55b"


def build_agent():
    # The backend is rooted at this folder (virtual paths: "/AGENTS.md",
    # "/skills/", "/output/..."), so memory and skills are loaded through it
    # and the agent's writes land in ./output/.
    return create_deep_agent(
        model=MODEL,
        memory=["/AGENTS.md"],
        skills=["/skills/"],
        backend=FilesystemBackend(root_dir=str(HERE), virtual_mode=True),
    )


def final_text(result) -> str:
    # Nemotron occasionally ends a tool-heavy turn with empty content, so
    # take the last non-empty assistant text from the run.
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "(no text reply — check output/ for files)"


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: uv run content_writer.py "your content request"')
        return 1

    agent = build_agent()
    result = agent.invoke({"messages": sys.argv[1]})
    print(final_text(result))

    output_dir = HERE / "output"
    if output_dir.exists():
        print("\nFiles in output/:")
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
