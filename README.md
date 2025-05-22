# Demo MCP Project

This repository demonstrates a multi-container setup using [FastAPI](https://fastapi.tiangolo.com/) and [FastMCP](https://github.com/huynhminhtufu/fastmcp) for building modular, composable API services. It includes two main services:

- **API Server**: A FastAPI-based hero management API, MCP-enabled for tool-based orchestration.
- **Other MCP Server**: A calculation microservice exposing date/datetime tools via FastMCP.

## Features

- **Hero Management API**: Create, list, retrieve, and delete heroes with rich metadata.
- **Calculation Tools**: Date/datetime difference, addition, and current time utilities.
- **MCP Integration**: Both services expose their APIs as MCP tools for agent-based workflows.
- **Dockerized**: Easily run both services with Docker Compose.
- **Jupyter Notebook**: Example notebook (`main.ipynb`) showing how to interact with the services and compose agents using [llama-index](https://github.com/jerryjliu/llama_index).

## Project Structure

```
.
├── api/                  # Hero management API (FastAPI + FastMCP)
│   ├── main.py
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── ...
├── other_mcp_server/     # Calculation service (FastMCP)
│   ├── main.py
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── ...
├── main.ipynb            # Example Jupyter notebook
├── compose.yaml          # Docker Compose configuration
└── README.md             # This file
```

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Running the Services

1. **Build and start all services:**

   ```sh
   docker compose up --build
   ```

   - The **API server** will be available at [http://localhost:8000](http://localhost:8000)
   - The **Other MCP server** will be available at [http://localhost:8001](http://localhost:8001)

### Development

- Code changes in `api/main.py` or `other_mcp_server/main.py` will automatically reload the respective service (see `compose.yaml`).

### Using the Notebook

- Install dependencies in your Python environment:

  ```sh
  pip install llama-index-tools-mcp llama-index python-dotenv llama-index-tools-wikipedia fastmcp
  ```

- Open `main.ipynb` in Jupyter and follow the examples to interact with the MCP-enabled services.

## Example MCP Tools

- **Hero API Tools**: Create, list, read, and delete heroes.
- **Calculation Tools**: `diff_datetime`, `add_datetime`, `get_current_time`.
