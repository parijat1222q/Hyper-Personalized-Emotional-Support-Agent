"""
Database client initialization and management.

This module handles:
  - Neo4j async driver initialization with retry logic
  - Redis connection pooling
  - Qdrant vector store client
  - HuggingFace InferenceClient setup
  - Startup and shutdown event handlers

All clients are initialized lazily and exposed as module-level variables
that are populated during FastAPI startup events.
"""

import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
import redis
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
import httpx


# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# HuggingFace Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:fastest")

# Neo4j Configuration
NEO4J_URL = os.getenv("NEO4J_URI", os.getenv("NEO4J_URL", "bolt://neo4j-graph:7687"))
NEO4J_AUTH = os.getenv("NEO4J_AUTH")

if not NEO4J_AUTH:
    raise ValueError("NEO4J_AUTH environment variable is required and must be set")

# Parse NEO4J_AUTH (format: "user/password")
if "/" in NEO4J_AUTH:
    NEO4J_USER, NEO4J_PASSWORD = NEO4J_AUTH.split("/", 1)
else:
    raise ValueError("NEO4J_AUTH must be in the format 'user/password'")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis-memory")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-vector")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# ============================================================================
# GLOBAL CLIENT INSTANCES
# ============================================================================

# Async clients (initialized in startup event)
neo4j_driver: Optional[AsyncGraphDatabase] = None
httpx_client: Optional[httpx.AsyncClient] = None

# Sync clients (initialized at module import time)
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None
hf_client: Optional[InferenceClient] = None


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

async def init_neo4j() -> Optional[AsyncGraphDatabase]:
    """
    Initialize Neo4j async driver with retry logic to handle docker-compose startup drift.
    
    Returns:
        AsyncGraphDatabase driver instance or None if initialization failed
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            driver = AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Verify connectivity with async call
            async with driver.session() as session:
                await session.run("RETURN 1")
            logger.info("✓ Neo4j async driver connected")
            return driver
        except Exception as e:
            logger.error(f"✗ Neo4j connection failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(4)
    return None


def init_redis() -> Optional[redis.Redis]:
    """
    Initialize Redis client with connection pooling.
    
    Returns:
        Redis client instance or None if initialization failed
    """
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        redis_client.ping()
        logger.info("✓ Redis connected")
        return redis_client
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        return None


def init_qdrant() -> Optional[QdrantClient]:
    """
    Initialize Qdrant vector store client.
    
    Returns:
        QdrantClient instance or None if initialization failed
    """
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("✓ Qdrant connected")
        return qdrant_client
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        return None


def init_hf_client() -> Optional[InferenceClient]:
    """
    Initialize HuggingFace InferenceClient for LLM calls.
    
    Returns:
        InferenceClient instance or None if initialization failed
    """
    try:
        if not HF_TOKEN:
            logger.warning("⚠ HF_TOKEN not set - using anonymous access")
            return InferenceClient(model=MODEL_NAME)
        return InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"✗ HF InferenceClient initialization failed: {e}")
        return None


# ============================================================================
# STARTUP AND SHUTDOWN HANDLERS
# ============================================================================

async def startup_clients() -> None:
    """
    Initialize all clients on FastAPI startup.
    This function should be called in @app.on_event("startup").
    """
    global neo4j_driver, httpx_client, redis_client, qdrant_client, hf_client

    logger.info("[DB Clients] Initializing all database and external clients...")

    # Initialize async clients
    neo4j_driver = await init_neo4j()
    httpx_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=100)
    )
    logger.info("[DB Clients] ✓ Async clients initialized (Neo4j, httpx)")

    # Initialize sync clients
    redis_client = init_redis()
    qdrant_client = init_qdrant()
    hf_client = init_hf_client()
    logger.info("[DB Clients] ✓ Sync clients initialized (Redis, Qdrant, HuggingFace)")


async def shutdown_clients() -> None:
    """
    Gracefully shutdown all clients on FastAPI shutdown.
    This function should be called in @app.on_event("shutdown").
    """
    global neo4j_driver, httpx_client, redis_client

    logger.info("[DB Clients] Shutting down all database and external clients...")

    # Close async clients
    if httpx_client:
        await httpx_client.aclose()
        logger.info("[DB Clients] ✓ HTTPX client closed")

    if neo4j_driver:
        await neo4j_driver.close()
        logger.info("[DB Clients] ✓ Neo4j async driver closed")

    # Close sync clients
    if redis_client:
        redis_client.close()
        logger.info("[DB Clients] ✓ Redis client closed")

    logger.info("[DB Clients] All clients shutdown complete")


# ============================================================================
# HELPER FUNCTIONS FOR CLIENT VERIFICATION
# ============================================================================

def all_clients_ready() -> bool:
    """Check if all critical clients are initialized."""
    return (
        neo4j_driver is not None
        and redis_client is not None
        and qdrant_client is not None
        and hf_client is not None
        and httpx_client is not None
    )


def get_client_status() -> dict:
    """Get status of all clients."""
    return {
        "neo4j": neo4j_driver is not None,
        "redis": redis_client is not None,
        "qdrant": qdrant_client is not None,
        "hf_client": hf_client is not None,
        "httpx_client": httpx_client is not None,
        "model": MODEL_NAME,
    }
