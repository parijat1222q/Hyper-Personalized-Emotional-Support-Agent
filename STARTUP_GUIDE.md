# OmniMind - Docker Startup Guide

## 1) Start Full Stack

```powershell
cd "g:\track b\omnimind"
docker-compose up -d --build
docker-compose ps
```

## 2) Service Health Checks

```powershell
# Python Worker (Port 5000)
curl.exe -I http://localhost:5000/docs

# Neo4j Database (Port 7474)
curl.exe http://localhost:7474/

# Go Gateway (Port 8080)
curl.exe http://localhost:8080/

# Node Orchestrator (Port 4000)
curl.exe -I http://localhost:4000/

# Next.js Frontend (Port 3000)
curl.exe -I http://localhost:3000/
```

## 3) Frontend Is Docker-First

- `frontend-nextjs` image builds dependencies inside Docker.
- Host-side `frontend-nextjs/node_modules` is not required for Docker build/run.
- Keep using:

```powershell
docker-compose build frontend-nextjs
docker-compose up -d frontend-nextjs
```

## 4) Quick Access

| Service | URL | Notes |
| --- | --- | --- |
| Neo4j Browser | http://localhost:7474 | `neo4j / omnipassword123` |
| Python Worker (Swagger) | http://localhost:5000/docs | - |
| Go Gateway | http://localhost:8080 | - |
| Node Orchestrator | http://localhost:4000 | - |
| Next.js Frontend | http://localhost:3000 | Cyberpunk UI |

## 5) Stop Services

```powershell
docker-compose down
```

