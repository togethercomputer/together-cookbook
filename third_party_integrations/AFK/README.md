# Use Together AI with AFK coding-agent sessions

[AFK](https://afk.mooglest.com) is a browser-based command center for persistent coding-agent sessions. AFK has a built-in Together AI connection, so you can bring your Together API key, pick a Together model per session, and supervise coding-agent work from the browser.

Use this integration when you want AFK agents to work on code, reviews, debugging, documentation, or longer-running development tasks with models hosted by Together AI.

## Prerequisites

- A Together AI API key from [api.together.ai](https://api.together.ai/)
- An AFK account at [afk.mooglest.com](https://afk.mooglest.com)
- An AFK daemon connected to the machine that has access to your project files

## 1. Create or sign in to AFK

Open [afk.mooglest.com](https://afk.mooglest.com) and create an account or sign in.

AFK runs from the browser UI while a daemon gives sessions access to your local or remote project directories.

## 2. Install and connect an AFK daemon

In AFK:

1. Open **Account → API Keys**.
2. Create a daemon token.
3. Follow the install command shown in the app.
4. Confirm the daemon appears as connected in the browser.

## 3. Add Together AI as an LLM connection

In AFK:

1. Open **Account → LLM**.
2. Click **Add connection**.
3. Choose **Together AI**.
4. Paste your Together API key.
5. Leave **Base URL** blank unless you are routing through a custom proxy or gateway.
6. Save or test the connection.

AFK uses Together AI's default OpenAI-compatible endpoint automatically for the built-in Together provider.

## 4. Start a session with a Together model

Click **New session** in AFK, then:

1. Select the connected daemon and project directory.
2. Choose the Together AI connection.
3. Select or type a Together model name, for example:

   ```text
   meta-llama/Llama-3.3-70B-Instruct-Turbo
   meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
   Qwen/Qwen2.5-72B-Instruct-Turbo
   ```

4. Choose a permission mode.
5. Enter the coding task and start the session.

AFK will route the session's model requests through Together AI while the browser UI shows progress, tool usage, diffs, and session history.

## Optional: use a proxy or gateway

If your team routes Together traffic through an internal gateway, set **Base URL** to the gateway's OpenAI-compatible endpoint.

For example:

```text
https://your-gateway.example.com/v1
```

Keep Base URL blank for normal Together AI usage.

## Troubleshooting

| Issue | Check |
|-------|-------|
| Connection test fails | Verify the Together API key and confirm your network can reach Together AI. |
| Model is missing | Manually type the Together model name in AFK. Provider model discovery can lag behind newly released models. |
| Custom gateway errors | Confirm the Base URL includes the OpenAI-compatible `/v1` path expected by your gateway. |
| Session cannot access files | Confirm the selected AFK daemon is connected and has the project directory under an allowed root. |

## Resources

- [AFK](https://afk.mooglest.com)
- [AFK provider setup docs](https://docs.mooglest.com/providers)
- [Together AI docs](https://docs.together.ai/docs/introduction)
- [Together AI models](https://docs.together.ai/docs/serverless-models)
