import { Agent } from "@mastra/core/agent";
import { weatherTool } from "./weather-tool.js";

// Mastra's model router speaks to Together AI directly: pass a `togetherai/`
// model string and it reads TOGETHER_API_KEY from the environment. No
// provider package to install.
export const weatherAgent = new Agent({
  id: "weather-agent",
  name: "Weather Agent",
  instructions: `
      You are a helpful weather assistant that provides accurate weather
      information and can help plan activities based on the weather.
      Use the weatherTool to fetch current weather data.
`,
  model: "togetherai/moonshotai/Kimi-K3",
  tools: { weatherTool },
});
