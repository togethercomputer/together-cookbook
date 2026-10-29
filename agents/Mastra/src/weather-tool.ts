import { createTool } from "@mastra/core/tools";
import { z } from "zod";

// Keyless weather lookup via Open-Meteo: geocode the city, then read the
// current conditions. Same shape as the tool create-mastra scaffolds.
export const weatherTool = createTool({
  id: "get-weather",
  description: "Get current weather for a location",
  inputSchema: z.object({
    location: z.string().describe("City name"),
  }),
  outputSchema: z.object({
    location: z.string(),
    temperatureC: z.number(),
    windSpeedKmh: z.number(),
  }),
  execute: async ({ location }) => {
    const geo = await fetch(
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1`,
    ).then((r) => r.json() as Promise<{ results?: { latitude: number; longitude: number; name: string }[] }>);
    const place = geo.results?.[0];
    if (!place) {
      throw new Error(`Location '${location}' not found`);
    }
    const forecast = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}&current=temperature_2m,wind_speed_10m`,
    ).then((r) => r.json() as Promise<{ current: { temperature_2m: number; wind_speed_10m: number } }>);
    return {
      location: place.name,
      temperatureC: forecast.current.temperature_2m,
      windSpeedKmh: forecast.current.wind_speed_10m,
    };
  },
});
