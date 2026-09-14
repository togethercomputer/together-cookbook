# Grafana dashboard for Together AI dedicated endpoints

An example [Grafana](https://grafana.com/) dashboard for monitoring [Together AI dedicated endpoints](https://docs.together.ai/docs/dedicated-endpoints/overview), built on the org-scoped Prometheus-compatible metrics endpoint.

The dashboard covers, per endpoint and deployment:

- Golden signals at the edge: request rate by status code, 5xx error ratio, and in-flight requests.
- Client-observed latency: request duration and time to first token percentiles (p50/p90/p99, milliseconds).
- Server-side latency: router and worker durations, time to first token, and time per output token (seconds).
- Throughput and tokens: request rate by layer, token throughput, and tokens per request.
- Engine health: KV cache utilization and prefix cache hit rate.

## Requirements

- Grafana 10.2 or later.
- A Prometheus-compatible datasource that scrapes `https://o11y-de2-metrics.cloud.together.ai/organizations/{org_id}/metrics` with your Together AI API key as a bearer token. The metrics endpoint is in beta, and access may need to be enabled for your organization.

## Usage

Follow the step-by-step guide in the Together AI docs: [Visualize endpoint metrics in Grafana](https://docs.together.ai/docs/dedicated-endpoints/grafana). In short:

1. Point a Prometheus scraper at the metrics endpoint (see the [monitoring reference](https://docs.together.ai/docs/dedicated-endpoints/monitoring#prometheus-compatible-metrics-endpoint) for the scrape config).
2. In Grafana, go to **Dashboards** > **New** > **Import**.
3. Upload [`together-dedicated-endpoints-dashboard.json`](./together-dedicated-endpoints-dashboard.json) and select your Prometheus datasource.

Use the **Endpoint** and **Deployment** variables at the top of the dashboard to filter.
