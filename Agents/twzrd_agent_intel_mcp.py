"""
TWZRD Agent Intel — MCP Trust Verification with Together AI
============================================================

Use Together AI models with TWZRD Agent Intel to verify agent trustworthiness
before making x402 payments. TWZRD provides a free MCP server at
https://intel.twzrd.xyz/mcp with two free tools and one paid tool.

Free tools:
  - score_agent(wallet)     → trust score (0-100) + risk signals
  - preflight_check(wallet) → quick go/no-go before payment

Paid tool (HTTP 402 / x402):
  - get_trust_receipt(wallet) → signed on-chain receipt

Setup:
  pip install together mcp

Docs: https://intel.twzrd.xyz
"""
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from together import Together

TWZRD_MCP = "https://intel.twzrd.xyz/mcp"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Example agent wallet from the x402 ecosystem (Dexter repeat payer)
EXAMPLE_WALLET = "D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi"


async def score_agent_trust(wallet: str) -> dict:
    """Call TWZRD MCP server to score an agent wallet."""
    async with streamablehttp_client(TWZRD_MCP) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            tools = [t.name for t in tools_result.tools]
            print(f"Available TWZRD tools: {tools}")

            # Score the agent
            result = await session.call_tool("score_agent", {"wallet": wallet})
            score_data = result.content[0].text

            # Run preflight check
            preflight = await session.call_tool("preflight_check", {"wallet": wallet})
            preflight_data = preflight.content[0].text

            return {"score": score_data, "preflight": preflight_data}


def analyze_trust_with_llm(wallet: str, trust_data: dict) -> str:
    """Use Together AI to analyze the trust data and produce a recommendation."""
    client = Together()

    system_prompt = (
        "You are a Web3 payment security advisor. "
        "Analyze agent trust scores and make clear pay/no-pay recommendations. "
        "Be concise — one paragraph max."
    )

    user_message = (
        f"Agent wallet: {wallet}\n\n"
        f"Trust score data:\n{trust_data['score']}\n\n"
        f"Preflight check:\n{trust_data['preflight']}\n\n"
        "Should I proceed with an x402 payment to this agent? Give a clear recommendation."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content


async def main():
    print(f"Checking trust for wallet: {EXAMPLE_WALLET}\n")

    # Step 1: Fetch trust data via MCP
    trust_data = await score_agent_trust(EXAMPLE_WALLET)
    print(f"Score data:\n{trust_data['score']}\n")
    print(f"Preflight:\n{trust_data['preflight']}\n")

    # Step 2: LLM analysis with Together AI
    print("Analyzing with Together AI...\n")
    recommendation = analyze_trust_with_llm(EXAMPLE_WALLET, trust_data)
    print(f"Recommendation:\n{recommendation}")


if __name__ == "__main__":
    asyncio.run(main())
