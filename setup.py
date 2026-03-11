from setuptools import setup, find_packages

setup(
    name="azure-langgraph-quickstart",
    version="0.1.0",
    description="LangGraph agent with multi-server MCP tool support on Azure",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2",
        "langchain-core>=0.2",
        "langchain-openai>=0.1",
        "langchain-mcp-adapters>=0.2",
        "openai>=1.30",
        "fastapi>=0.111",
        "uvicorn[standard]>=0.29",
        "pyyaml>=6.0",
        "httpx>=0.27",
        "typing-extensions>=4.11",
    ],
    extras_require={
        "dev": [
            "pytest>=8",
            "pytest-asyncio>=0.23",
            "httpx",
            "ruff>=0.4",
            "mypy>=1.10",
        ]
    },
)
