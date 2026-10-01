import { weatherAgent } from "./weather-agent.js";

// Kimi K3 reasons by default. Turn reasoning off for this call by passing
// provider options through generate().
const response = await weatherAgent.generate(
  "What's the weather in San Francisco today?",
  {
    providerOptions: {
      togetherai: { reasoning: { enabled: false } },
    },
  },
);

console.log(response.text);
