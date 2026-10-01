# Mastra weather agent

Runnable companion to the [Mastra quickstart](https://docs.together.ai/docs/using-together-with-mastra) in the Together AI docs. The guide explains how to point a Mastra agent at Together AI, and this folder runs it end to end: a weather agent with one tool, served by a Together AI serverless model through Mastra's model router.

## What it demonstrates

- Passing a `togetherai/` model string to a Mastra `Agent`, so Mastra reads `TOGETHER_API_KEY` from the environment with no provider package to install.
- A `createTool` tool (Open-Meteo, keyless) the agent calls before answering.
- Turning reasoning off for a call on a model that reasons by default (Kimi K3) with `providerOptions.togetherai.reasoning`.

## Run it

Requires Node 22 or later and a Together AI API key.

```bash
npm install
export TOGETHER_API_KEY=your_api_key
npm start
```

`npm run typecheck` runs the TypeScript compiler without emitting files. Execution CI runs both commands weekly (see `ci.yaml`).

## Files

| File | What it is |
| --- | --- |
| `src/weather-agent.ts` | The agent from the guide, pointed at `togetherai/moonshotai/Kimi-K3`. |
| `src/weather-tool.ts` | The weather tool the agent calls. |
| `src/index.ts` | Generates one response with reasoning disabled and prints it. |

The model ID matches the guide. If the guide changes models, change it here too.
