# OmniMind — Empathetic Mental Health AI

> **Track B Competition Submission** · A stateful, multi-layered conversational AI that goes far beyond basic RAG. Built on an "antigravity" microservices philosophy — fluid, effortless, and infinitely scalable.

---

## Architecture Overview

OmniMind implements a mandatory **4-Layer AI Architecture** designed to read between the lines of human conversation, maintain long-term episodic memory, and generate empathetic, hyper-personalized responses.

```
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 1 · Message Intake & Analysis              │
│   Go API Gateway (Omni-Input Handler + Request Routing)         │
│                             ↓                                   │
│   Node.js Orchestrator (Multi-Turn Entity Resolution +          │
│           Safety & Risk Escalation Guardrails)                  │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 2 · Temporal Knowledge Graph & Memory           │
│   Redis (Short-Term Working Memory & Session Queue)             │
│   Neo4j (Long-Term Episodic/Semantic Knowledge Graph)           │
│           Stores facts as semantic triples with validity windows │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│        LAYER 3 · Dynamic Knowledge Pipeline & Retrieval         │
│   Python FastAPI Worker (Gemma 4 via Hugging Face Serverless)   │
│   Memory Distillation Endpoint → Neo4j Cypher MERGE Injection   │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│       LAYER 4 · Contextual Response Generation (Upcoming)       │
│   Next.js Cyberpunk/Sci-Fi Conversational UI                    │
│   State Injection → Targeted Prompt Engineering → Response Check│
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **API Gateway** | Go (Golang) | High-performance frictionless ingress, concurrent request routing |
| **Orchestrator** | Node.js + TypeScript | Central nervous system, safety guardrails, entity resolution |
| **AI Worker** | Python 3.11 + FastAPI | LLM extraction, memory distillation, Neo4j graph injection |
| **LLM** | Gemma 4 (`google/gemma-4-E4B-Instruct`) | Semantic triple extraction via Hugging Face Serverless API |
| **Knowledge Graph** | Neo4j 5.26 | Long-term episodic & semantic memory (Temporal Knowledge Graph) |
| **Working Memory** | Redis 7 | Short-term session context and inter-service message queuing |
| **Frontend** | Next.js + React | Cyberpunk/Sci-Fi conversational interface *(upcoming)* |
| **Infrastructure** | Docker + Docker Compose | Fully containerized microservices on the `omninet` bridge network |

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker Desktop** (required) — running with WSL2 or Hyper-V backend recommended
- **Git**
- A **Hugging Face account** with a valid API token ([get one here](https://huggingface.co/settings/tokens))

---

## Setup & Configuration

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd omnimind
```

### 2. Configure Environment Variables

Copy the `.env` file and fill in your secrets. **Never commit your real token to Git.**

```bash
# .env — edit this file before running docker-compose
NEO4J_AUTH=neo4j/omnipassword123
REDIS_PORT=6379
NODE_PORT=4000
GATEWAY_PORT=8080

# 🔑 Replace with your actual Hugging Face token
HF_TOKEN=hf_your_huggingface_token
HF_MODEL_NAME=google/gemma-4-E4B-Instruct
```

> **⚠️ Important**: The `HF_TOKEN` is required for the AI Worker to call Gemma 4. Without it, the `/api/memory/distill` endpoint will return a 502 error.

---

## Quickstart

### Build and Launch the Full Network

Open a terminal in the root `omnimind/` directory and run:

```bash
docker-compose up -d --build
```

This single command will:
- Pull the Neo4j, Redis, and base OS images
- Build the custom Go gateway, Node.js orchestrator, and Python AI worker containers
- Create the `omninet` internal bridge network
- Mount persistent volumes for Neo4j and Redis data

---

## Service Access Points

Once all containers are healthy, access each layer via:

| Service | URL | Notes |
|---|---|---|
| **Neo4j Dashboard** | [http://localhost:7474](http://localhost:7474) | Username: `neo4j` / Password: `omnipassword123` |
| **AI Worker Swagger UI** | [http://localhost:5000/docs](http://localhost:5000/docs) | Interactive endpoint testing for `/api/memory/distill` |
| **Go API Gateway** | `http://localhost:8080/api/intake` | `POST` — Omni-Input Handler |
| **Node.js Orchestrator** | `http://localhost:4000/api/analyze` | `POST` — Safety & Entity Pipeline |

---

## Testing the Memory Distillation Pipeline

Send a test request to the full pipeline using `curl` or the Swagger UI at `http://localhost:5000/docs`:

```bash
curl -X POST http://localhost:5000/api/memory/distill \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "session_id": "sess_abc123",
    "resolved_text": "I have been feeling really overwhelmed with my workload lately and I cannot sleep."
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Dynamic Memory successfully distilled and embedded into graph.",
  "extracted_triples": [
    { "subject": "User", "predicate": "STRUGGLING_WITH", "object": "Work Overload" },
    { "subject": "User", "predicate": "EXPERIENCING", "object": "Sleep Disruption" }
  ]
}
```

You can then visualize these nodes and relationships live inside the **Neo4j Browser** at `http://localhost:7474` by running:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

---

## Project Structure

```
omnimind/
├── .env                         # 🔐 Environment secrets (never commit)
├── .gitignore
├── docker-compose.yml           # Full service orchestration
│
├── gateway-go/                  # Layer 1: Go API Gateway
│   ├── cmd/server/main.go       # Omni-Input Handler
│   ├── Dockerfile               # Multi-stage optimized build
│   └── go.mod
│
├── orchestrator-node/           # Layer 1: Node.js Safety & Orchestration
│   ├── src/index.ts             # Risk Guardrails + Entity Resolution
│   ├── Dockerfile
│   ├── package.json
│   └── tsconfig.json
│
├── ai-python/                   # Layer 3: Python AI Memory Worker
│   ├── app/main.py              # FastAPI + Gemma 4 + Neo4j Cypher injection
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend-nextjs/             # Layer 4: Cyberpunk UI (upcoming)
└── docker/                      # Auxiliary Docker configs
```

---

## Internal Docker Network

All containers communicate over the `omninet` bridge network using their service names as hostnames:

```
gateway-go ──→ orchestrator-node:4000
orchestrator-node ──→ redis-memory:6379
ai-python ──→ neo4j-graph:7687
ai-python ──→ redis-memory:6379
```

Persistent data is stored in named Docker volumes:
- `omnimind_neo4j_data` — Knowledge Graph nodes and relationships
- `omnimind_redis_data` — Working memory and queued sessions

---

## Stopping the System

To gracefully shut down all containers while preserving your data volumes:

```bash
docker-compose down
```

To **also wipe** all stored Neo4j graph data and Redis memory (full reset):

```bash
docker-compose down -v
```

---

## Roadmap

- [x] Phase 1 — Go API Gateway & Node.js Orchestrator + Safety Guardrails
- [x] Phase 2 — Docker infrastructure (Neo4j, Redis, Bridge Network, Volumes)
- [x] Phase 3 — Python AI Worker (FastAPI + Neo4j Cypher integration)
- [x] Phase 4 — Gemma 4 LLM Extraction via Hugging Face Serverless API
- [ ] Phase 5 — Hybrid Search Engine (Dense Vector + BM25 + Graph Traversal)
- [ ] Phase 6 — Next.js Cyberpunk/Sci-Fi Conversational Frontend
- [ ] Phase 7 — Response Verification Layer (lightweight hallucination checker)

---

## License

This project is developed as a Track B competition submission.
See [LICENSE](LICENSE) for details.
