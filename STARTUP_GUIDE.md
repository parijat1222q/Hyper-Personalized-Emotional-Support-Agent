# OmniMind - Restart


---

cd "g:\track b\omnimind"
docker ps -a
docker-compose up -d

docker-compose up -d --build



# Python Worker (Port 5000)
curl.exe -I http://localhost:5000/docs

# Neo4j Database (Port 7474)
curl.exe http://localhost:7474/

# Go Gateway (Port 8080)
curl.exe http://localhost:8080/

# Node Orchestrator (Port 4000)
curl.exe -I http://localhost:4000/


| **Neo4j Browser** | http://localhost:7474 | neo4j / omnipassword123 |
| **Python Worker (Swagger)** | http://localhost:5000/docs | - |
| **Go Gateway** | http://localhost:8080 | - |
| **Node Orchestrator** | http://localhost:4000 | - |

---

