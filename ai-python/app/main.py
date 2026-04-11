from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
import redis
import os
import json
import logging
from openai import OpenAI

app = FastAPI(title="OmniMind AI Worker", version="1.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core Environment Variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/omnipassword123")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Gemma 4 / Hugging Face Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/gemma-4-E4B-it")

# Initialize OpenAI wrapper pointing to Hugging Face Serverless API
llm_client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1",
    api_key=HF_TOKEN
)

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
    # GENUINE LLM EXTRACTION (Gemma 4 via Hugging Face)
    # -------------------------------------------------------------
    system_prompt = """
    You are an expert clinical entity extractor. Your sole job is to interpret the user's expression and map it into semantic triples representing mental health states, life events, or behaviors.
    
    Constraint 1: Output MUST be a strictly valid JSON array of objects.
    Constraint 2: Every object MUST contain exactly these keys: "subject", "predicate", "object".
    Constraint 3: Create predicates in uppercase with underscores (e.g., EXPRESSED_CONCERN, STRUGGLING_WITH, FEELS_EMOTION).
    Constraint 4: Do not include markdown codeblocks or thought processes in the output. ONLY return the JSON array.
    """
    
    user_prompt = f"Analyze the following text and extract the core mental health semantic triples: {request.resolved_text}"
    
    try:
        response = llm_client.chat.completions.create(
            model=HF_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Clean up possible markdown code blocks from the LLM
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()
            
        triples = json.loads(raw_output)
        
        if not isinstance(triples, list):
            raise ValueError("LLM did not return a JSON array.")
            
    except json.JSONDecodeError:
        logger.error(f"Failed to decode LLM JSON Output: {raw_output}")
        raise HTTPException(status_code=500, detail="LLM failed to adhere to JSON format constraints.")
    except Exception as e:
        logger.error(f"LLM API Call Error: {e}")
        raise HTTPException(status_code=502, detail="Hugging Face API Unreachable or Timed Out.")

    # -------------------------------------------------------------
    # DYNAMIC NEO4J MERGE
    # -------------------------------------------------------------
    valid_merges = []
    
    cypher_query = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $object})
    // Dynamic relationship creation requires APOC or splitting the query. 
    // We use a generalized structure or string formatting carefully to insert dynamic predicates.
    WITH s, o
    CALL apoc.create.relationship(s, $predicate, {session_id: $session_id}, o) YIELD rel
    RETURN s, rel, o
    """
    # NOTE: Neo4j does not natively allow parameterizing relationship TYPES. 
    # To fix this safely without risking Cypher Injection, we format it after sanitizing the predicate string.
    
    with neo4j_driver.session() as session:
        for triple in triples:
            subject = triple.get("subject")
            predicate = triple.get("predicate", "").upper().replace(" ", "_").replace("-", "_")
            obj = triple.get("object")
            
            # Simple sanitization filter for alphanumeric + underscore on predicate
            clean_predicate = "".join([c for c in predicate if c.isalnum() or c == "_"])
            if not clean_predicate or not subject or not obj:
                continue

            # Safe string formatting for Relationship Type
            dynamic_cypher = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Concept {{name: $object}})
            MERGE (s)-[r:{clean_predicate} {{session_id: $session_id}}]->(o)
            RETURN s, r, o
            """
            
            try:
                session.run(
                    dynamic_cypher, 
                    subject=subject,
                    object=obj,
                    session_id=request.session_id
                ).consume()
                valid_merges.append(triple)
            except Exception as e:
                logger.error(f"Failed to merge triple {triple}: {e}")
                # We log but continue trying the rest
                continue
                
    logger.info(f"Successfully processed {len(valid_merges)} triples into Neo4j graph.")

    return {
        "status": "success",
        "message": "Dynamic Memory successfully distilled and embedded into graph.",
        "extracted_triples": valid_merges
    }
