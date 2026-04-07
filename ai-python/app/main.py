from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
import redis
import os
import logging

app = FastAPI(title="OmniMind AI Worker", version="1.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/omnipassword123")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Neo4j Driver Setup
neo_user, neo_pass = NEO4J_AUTH.split('/')
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(neo_user, neo_pass))

# Redis Client Setup
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

class DistillRequest(BaseModel):
    user_id: str
    session_id: str
    resolved_text: str

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Python Worker - Verifying Connections")
    try:
        neo4j_driver.verify_connectivity()
        logger.info("Connected to Neo4j Successfully!")
    except Exception as e:
        logger.error(f"Neo4j Connection Failed: {e}")

    try:
        if redis_client.ping():
            logger.info("Connected to Redis Successfully!")
    except Exception as e:
        logger.error(f"Redis Connection Failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    neo4j_driver.close()
    redis_client.close()

@app.post("/api/memory/distill")
async def extract_memory(request: DistillRequest):
    logger.info(f"Distilling memory for User: {request.user_id}")
    
    # -------------------------------------------------------------
    # MOCK LLM EXTRACTION
    # In production, we pass 'resolved_text' to DeepSeek/Llama
    # to extract subjects, predicates, and object semantic triples.
    # -------------------------------------------------------------
    mock_triple = {
        "subject": request.user_id,
        "predicate": "EXPRESSED_CONCERN",
        "object": "Work Stress"
    }
    
    # Neo4j Graph Cypher Merge Context (MERGE ensures no duplicates)
    cypher_query = """
    MERGE (u:User {id: $subject})
    MERGE (c:Concept {name: $object})
    MERGE (u)-[r:EXPRESSED_CONCERN {session_id: $session_id}]->(c)
    RETURN u, r, c
    """
    
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                cypher_query, 
                subject=mock_triple["subject"],
                object=mock_triple["object"],
                session_id=request.session_id
            )
            # Fetch summary
            result.consume()
        
        logger.info(f"Successfully merged semantic triple into Neo4j: {mock_triple}")
        
    except Exception as e:
        logger.error(f"Cypher Transaction Error: {e}")
        raise HTTPException(status_code=500, detail="Graph Database Transaction Failed")

    return {
        "status": "success",
        "message": "Memory successfully distilled and embedded into graph.",
        "extracted_triple": mock_triple
    }
