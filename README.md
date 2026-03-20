docker build -t my_agent .
docker run -d --name my_agent_run --network=host --env-file .env \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  my_agent:latest