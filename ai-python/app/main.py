"""
OmniMind AI Worker - Main Entry Point

Production-hardened FastAPI application with:
  - Async-first architecture (AsyncGraphDatabase, httpx.AsyncClient)
  - Modular design (routes, services, models, db clients)
  - Health checks and graceful startup/shutdown
  - Comprehensive logging and error handling
  - Docker-ready deployment
"""

import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.clients import startup_clients, shutdown_clients

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="OmniMind AI Worker",
    description="Empathetic AI mental health support system with hybrid memory retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to known origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database clients and perform health checks"""
    logger.info("=" * 80)
    logger.info("🚀 OmniMind AI Worker Starting Up")
    logger.info("=" * 80)
    
    try:
        await startup_clients()
        logger.info("✓ All clients initialized successfully")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close database connections"""
    logger.info("=" * 80)
    logger.info("🔌 OmniMind AI Worker Shutting Down")
    logger.info("=" * 80)
    
    try:
        await shutdown_clients()
        logger.info("✓ All clients closed gracefully")
    except Exception as e:
        logger.error(f"✗ Shutdown error: {e}")


# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

app.include_router(router, prefix="", tags=["endpoints"])
logger.info("✓ Routes registered")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting OmniMind AI Worker with uvicorn...")
    
    uvicorn.run(
        "app.main:app",  # Updated to reflect modular structure
        host="0.0.0.0",
        port=5000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )