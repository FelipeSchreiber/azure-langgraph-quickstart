# Telemetry Backend

Minimal OpenTelemetry backend for collecting telemetry data from MCP Server.

## Components

- **OpenTelemetry Collector**: Receives, processes, and exports telemetry data
- **Jaeger**: Distributed tracing visualization
- **Prometheus**: Metrics storage and querying

## Quick Start

```bash
# Start the telemetry backend
docker compose up -d

# View logs
docker compose logs -f

# Stop the backend
docker compose down
```

## Access Points

- **Jaeger UI**: http://localhost:16686
- **Prometheus UI**: http://localhost:9090
- **OTEL Collector Health**: http://localhost:13133

## OTLP Endpoints

The MCP Server should send telemetry to:
- **gRPC**: `http://localhost:4317`
- **HTTP**: `http://localhost:4318`

## MCP Server Configuration

Update your MCP Server's `.env` file:

```env
ENABLE_OTEL=true
OTEL_SERVICE_NAME=mcp-server
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Or if MCP Server is in Docker, use host networking:

```env
ENABLE_OTEL=true
OTEL_SERVICE_NAME=mcp-server
OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317
```

## Viewing Traces

1. Open Jaeger UI: http://localhost:16686
2. Select service: `mcp-server`
3. Click "Find Traces"

## Viewing Metrics

1. Open Prometheus UI: http://localhost:9090
2. Query metrics like: `up`, `http_server_duration_milliseconds`, etc.

## Architecture

```
MCP Server --> OTEL Collector --> Jaeger (traces)
                              --> Prometheus (metrics)
```

## Troubleshooting

Check collector logs:
```bash
docker compose logs otel-collector
```

Check collector health:
```bash
curl http://localhost:13133
```

## Data Persistence

- Jaeger uses in-memory storage (data lost on restart)
- Prometheus uses volume `prometheus-data` (persisted)

To reset Prometheus data:
```bash
docker compose down -v
```
