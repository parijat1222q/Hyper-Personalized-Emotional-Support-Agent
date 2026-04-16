"""
OmniMind AI Worker - Database Clients Package

Export all database client instances and initialization functions.
"""

from .clients import (
    # Configuration
    HF_TOKEN,
    MODEL_NAME,
    NEO4J_URL,
    NEO4J_USER,
    NEO4J_PASSWORD,
    REDIS_HOST,
    REDIS_PORT,
    QDRANT_HOST,
    QDRANT_PORT,
    # Client Instances
    neo4j_driver,
    redis_client,
    qdrant_client,
    hf_client,
    httpx_client,
    # Initialization Functions
    init_neo4j,
    init_redis,
    init_qdrant,
    init_hf_client,
    startup_clients,
    shutdown_clients,
    all_clients_ready,
    get_client_status,
)

__all__ = [
    # Configuration
    "HF_TOKEN",
    "MODEL_NAME",
    "NEO4J_URL",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "REDIS_HOST",
    "REDIS_PORT",
    "QDRANT_HOST",
    "QDRANT_PORT",
    # Client Instances
    "neo4j_driver",
    "redis_client",
    "qdrant_client",
    "hf_client",
    "httpx_client",
    # Functions
    "init_neo4j",
    "init_redis",
    "init_qdrant",
    "init_hf_client",
    "startup_clients",
    "shutdown_clients",
    "all_clients_ready",
    "get_client_status",
]
