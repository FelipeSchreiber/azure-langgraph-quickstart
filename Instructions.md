based on ../gemini-fullstack-langgraph-quickstart I want you to scaffold a simple langgraph agent. 
It must receive a agent_config.yaml containing:
    - Agent ID
    - Agent Name
    - Deployment
    - Instructions
    - Agent Description
    - Temperature
    - Top P
    - mcp:
    servers:
    - name: 
        description:
        transport:
        - type: ""
        - endpoint: ""
        authentication:
        - type: ""
        - type: "bearer"
        - token_env_var: "NEWS_MCP_TOKEN"

    metadata:
    owner: ""
    version: ""
    created_at: ""

The project structure must be the following:

src/
- graph.py: contains the graph with multi server MCP Client. The nodes are START, END, llm_call, call_tool.
- app.py: simple fastAPI containing basic routes, such as '/chat', '/metrics', '/health/live', '/health/ready', with swagger as well
- config.py: containing help scripts to read the yaml and returning the configs

scripts/
example.sh: contains example on how to run a curl code to '/chat' asking to add two numbers

Dockerfile
