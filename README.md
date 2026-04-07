# OmniMind AI - Empathetic Mental Health Architecture

This project is a containerized, 4-layer stateful conversational AI architecture for mental health mapping and analysis. It utilizes Go, Node.js, Python (FastAPI), Neo4j, Redis, and an internal Ollama container running Google's Gemma 4.

## Prerequisites
Before you start, ensure you have:
1. Git installed.
2. **Docker Desktop** installed and running on your machine.
3. (Optional but Recommended) WSL2 or Hyper-V enabled for Docker.

## Quickstart Guide

If you just cloned this repository, follow these precise steps to bring the entire AI brain online:

### 1. Build and Start the Infrastructure
Open your terminal in the root directory (where the `docker-compose.yml` file is located) and run:
```bash
docker-compose up -d --build
```
*This command pulls the necessary images, builds the custom Go/Node/Python containers, and maps the internal networking/storage volumes.*

### 2. Download the Gemma 4 Model
Because the AI model is locally hosted for privacy and zero latency, you must pull the model weights into your container on the very first run. Execute:
```bash
docker exec -it omnimind-ollama ollama run gemma4
```
*Wait for the download to complete. Once you see the interactive prompt (`>>>`), you can type `/bye` to exit. The model is now cached in your persistent volume!*

## Localhost Access Points
Once the containers are running, you can access the various layers of the architecture via your browser or API clients (like Postman):

- **Knowledge Graph (Neo4j)**: [http://localhost:7474](http://localhost:7474)
  - Connect via the browser.
  - **Username**: `neo4j`
  - **Password**: `omnipassword123`
- **Memory Distillation API (Python / AI Worker)**: [http://localhost:5000/docs](http://localhost:5000/docs)
  - Access the built-in Swagger UI to test the LLM extraction endpoint directly.
- **API Gateway (Go)**: `http://localhost:8080/api/intake` (POST)
- **Central Orchestrator (Node.js)**: `http://localhost:4000/api/analyze` (POST)

## Stopping the System
When you are done testing, you can gracefully shut down the network by running:
```bash
docker-compose down
```
*Note: Your Neo4j graph data, Redis queues, and Ollama AI models will be saved automatically via Docker volumes.*
