"""
Pydantic models for request/response validation across all API endpoints.

This module contains all request and response schemas used by the OmniMind AI Worker.
Organized by feature:
  - Memory Distillation (Phase 3)
  - Knowledge Fetching (Phase 4)
  - Hybrid Retrieval (Phase 6)
  - Response Generation (Phase 7)
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ============================================================================
# MEMORY DISTILLATION (Phase 3)
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request for semantic triple extraction and Neo4j storage"""
    user_id: str
    resolved_text: str
    context: Optional[str] = None


class SemanticTriple(BaseModel):
    """Individual subject-predicate-object triple"""
    subject: str
    predicate: str
    object: str


class AnalyzeResponse(BaseModel):
    """Response from triple extraction endpoint"""
    status: str
    extracted_triples: List[Dict[str, str]]
    error: Optional[str] = None


# ============================================================================
# KNOWLEDGE FETCHING (Phase 4)
# ============================================================================

class KnowledgeFetchRequest(BaseModel):
    """Request for fetching external knowledge (Reddit, etc.)"""
    query: str
    source: str = "reddit"
    limit: int = 5


class KnowledgeFetchResponse(BaseModel):
    """Response from knowledge fetch endpoint"""
    status: str
    fetched_items: List[Dict[str, Any]]
    extracted_triples: List[Dict[str, str]]
    error: Optional[str] = None


# ============================================================================
# HYBRID RETRIEVAL (Phase 6)
# ============================================================================

class MemoryRetrieveRequest(BaseModel):
    """Request for hybrid memory retrieval (Vector + BM25 + Graph)"""
    user_id: str
    query: str
    top_k: int = 3


class FusedSearchResult(BaseModel):
    """Individual search result from RRF fusion"""
    id: str
    source: str  # "vector", "keyword", "graph"
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]


class MemoryRetrieveResponse(BaseModel):
    """Response from hybrid retrieval endpoint"""
    status: str
    fused_results: List[FusedSearchResult]
    context_payload: Dict[str, Any]
    error: Optional[str] = None


# ============================================================================
# RESPONSE GENERATION (Phase 7)
# ============================================================================

class GenerateRequest(BaseModel):
    """Request for empathetic response generation"""
    user_id: str
    query: str
    context: Optional[Dict[str, Any]] = None
    tone: str = "empathetic"  # empathetic, supportive, analytical, motivational
    include_citations: bool = True


class GeneratedResponse(BaseModel):
    """Generated response with confidence and citations"""
    response_text: str
    confidence: float
    citations: List[Dict[str, str]] = []
    generated_at: str = ""
    tone_used: str = ""


class GenerateResponse(BaseModel):
    """Response from response generation endpoint"""
    status: str
    user_id: str
    query: str
    generated_response: GeneratedResponse
    response_id: Optional[str] = None
    error: Optional[str] = None
