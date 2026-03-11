docker build -t my_agent .
docker run -d --name my_agent_run --network=host --env-file .env my_agent:latest 