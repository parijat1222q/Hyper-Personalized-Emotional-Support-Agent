"""
Business Logic and Core Services

This module contains all backend operations:
  - LLM semantic triple extraction
  - Neo4j storage and graph traversal
  - Redis caching
  - Hybrid search (Vector + BM25 + Graph)
  - Response generation with hallucination checking
  - External data fetching (Reddit)
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

import httpx

from app.db import clients
from app.core.hallucination import validate_and_repair_response

logger = logging.getLogger(__name__)


# ============================================================================
# LLM SEMANTIC TRIPLE EXTRACTION
# ============================================================================

def extract_semantic_triples_lvm(text: str) -> List[Dict[str, str]]:
    """Extract semantic triples from text using LLM"""
    if not clients.hf_client:
        logger.error("✗ HF Client not available")
        raise RuntimeError("HF Client not initialized for semantic triple extraction")
    
    try:
        system_prompt = """You are a semantic triple extraction expert for mental health conversations.
Extract subject-predicate-object triples from the given text. Return ONLY valid JSON array format.
Format: [{"subject": "...", "predicate": "...", "object": "..."}, ...]
Focus on: emotions, behaviors, triggers, symptoms, relationships, and experiences."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract triples from: {text}"}
        ]
        
        response = clients.hf_client.chat_completion(
            messages=messages,
            model=clients.MODEL_NAME,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Handle None response
        if not response_text:
            logger.error("✗ LLM returned empty response")
            raise ValueError("LLM returned empty response for triple extraction")
        
        # Parse JSON from response using regex
        match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(0)
            try:
                triples = json.loads(json_str)
                if isinstance(triples, list):
                    return triples[:10]
            except json.JSONDecodeError as je:
                logger.error(f"✗ Failed to parse extracted JSON: {je}")
                raise ValueError(f"JSON parse error: {je}")
        else:
            logger.error(f"✗ Failed to find JSON array in LLM response: {response_text}")
            raise ValueError("LLM response does not contain valid JSON array")
        
    except Exception as e:
        logger.error(f"✗ LLM extraction error: {e}")
        raise


# ============================================================================
# NEO4J STORAGE
# ============================================================================

async def store_triples_neo4j(user_id: str, triples: List[Dict[str, str]]) -> bool:
    """Store semantic triples in Neo4j using async driver"""
    if not clients.neo4j_driver:
        logger.warning("⚠ Neo4j not available")
        return False
    
    try:
        async with clients.neo4j_driver.session() as session:
            for triple in triples:
                query = """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATIONSHIP {
                    type: $predicate,
                    user_id: $user_id,
                    timestamp: datetime()
                }]->(o)
                RETURN r
                """
                
                await session.run(
                    query,
                    subject=triple.get("subject", "Unknown"),
                    predicate=triple.get("predicate", "UNKNOWN"),
                    object=triple.get("object", "Unknown"),
                    user_id=user_id
                )
            
            logger.info(f"✓ Stored {len(triples)} triples for user {user_id}")
            return True
            
    except Exception as e:
        logger.error(f"✗ Neo4j storage error: {e}")
        return False


# ============================================================================
# REDIS CACHING
# ============================================================================

def cache_analysis_redis(user_id: str, analysis: Dict[str, Any]) -> bool:
    """Cache analysis results in Redis"""
    if not clients.redis_client:
        logger.warning("⚠ Redis not available")
        return False
    
    try:
        cache_key = f"analysis:{user_id}:{datetime.now().isoformat()}"
        clients.redis_client.setex(
            cache_key,
            3600,  # 1 hour expiry
            json.dumps(analysis)
        )
        logger.info(f"✓ Cached analysis for {user_id}")
        return True
    except Exception as e:
        logger.error(f"✗ Redis caching error: {e}")
        return False


# ============================================================================
# EXTERNAL DATA FETCHING
# ============================================================================

async def fetch_reddit_data(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch public data from Reddit search API using async httpx"""
    if not clients.httpx_client:
        logger.error("✗ HTTPX client not initialized")
        return []
    
    try:
        logger.info(f"Fetching Reddit posts for: {query}")
        
        headers = {
            "User-Agent": "OmniMind-AI-Mental-Health/1.0 (Educational Purpose)"
        }
        
        # Use Reddit search JSON endpoint
        url = f"https://www.reddit.com/search.json"
        params = {
            "q": query,
            "sort": "new",
            "limit": limit,
            "type": "link"
        }
        
        # Non-blocking async request
        response = await clients.httpx_client.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        
        data = response.json()
        items = []
        
        if "data" in data and "children" in data["data"]:
            for post in data["data"]["children"][:limit]:
                post_data = post.get("data", {})
                items.append({
                    "id": post_data.get("id", ""),
                    "title": post_data.get("title", ""),
                    "text": post_data.get("selftext", ""),
                    "url": post_data.get("url", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "score": post_data.get("score", 0),
                    "created_utc": post_data.get("created_utc", 0)
                })
        
        logger.info(f"✓ Fetched {len(items)} Reddit items")
        return items
        
    except httpx.RequestError as e:
        logger.error(f"✗ Reddit fetch request error: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"✗ Reddit JSON parse error: {e}")
        return []
    except Exception as e:
        logger.error(f"✗ Unexpected error fetching Reddit: {e}")
        return []


# ============================================================================
# HYBRID SEARCH - RECIPROCAL RANK FUSION
# ============================================================================

def reciprocal_rank_fusion(results_dict: Dict[str, List[tuple]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) combines results from multiple search methods
    Formula: RRF(d) = sum over search methods of (1 / (k + rank(d)))
    
    Args:
        results_dict: {"vector": [(id, score, data)], "keyword": [(id, score, data)], "graph": [(id, score, data)]}
        k: RRF parameter (typically 60)
    
    Returns:
        List of combined results sorted by RRF score
    """
    rrf_scores = {}
    id_to_data = {}
    
    # Calculate RRF scores for each search method
    for method, results in results_dict.items():
        for rank, (result_id, score, data) in enumerate(results, start=1):
            rrf_component = 1.0 / (k + rank)
            
            if result_id not in rrf_scores:
                rrf_scores[result_id] = 0.0
                id_to_data[result_id] = {"sources": {}, "data": data}
            
            rrf_scores[result_id] += rrf_component
            id_to_data[result_id]["sources"][method] = {"rank": rank, "score": score}
    
    # Sort by RRF score and return
    fused_results = [
        (result_id, rrf_scores[result_id], id_to_data[result_id])
        for result_id in sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    ]
    
    return fused_results


# ============================================================================
# HYBRID SEARCH - VECTOR SEARCH
# ============================================================================

def search_vector_qdrant(query: str, limit: int = 3) -> List[tuple]:
    """
    Dense vector search using FastEmbed embeddings in Qdrant
    Returns: [(id, score, data), ...]
    """
    if not clients.qdrant_client:
        logger.warning("⚠ Qdrant not available for vector search")
        return []
    
    try:
        from fastembed import TextEmbedding
        
        logger.info(f"🔍 Vector search in Qdrant for: {query}")
        
        # Embed query
        embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        query_embedding = list(embedding_model.embed(query))[0]
        
        # Check which search method is available
        results = None
        try:
            # Try search_points (newer API)
            results_obj = clients.qdrant_client.search_points(
                collection_name="knowledge_base",
                query_vector=query_embedding,
                limit=limit
            )
            results = results_obj.points if hasattr(results_obj, 'points') else results_obj
        except (AttributeError, TypeError):
            try:
                # Try search method
                results = clients.qdrant_client.search(
                    collection_name="knowledge_base",
                    query_vector=query_embedding,
                    limit=limit
                )
            except (AttributeError, TypeError):
                logger.warning("⚠ No search method available on Qdrant client")
                return []
        
        if not results:
            logger.info("ℹ Vector search returned no results")
            return []
        
        vector_results = []
        for result in results:
            vector_results.append((
                str(result.id),
                float(result.score),
                {
                    "title": result.payload.get("name", "Unknown"),
                    "content": result.payload.get("instructions", ""),
                    "category": result.payload.get("category", "General"),
                    "effectiveness": result.payload.get("effectiveness", "")
                }
            ))
        
        logger.info(f"✅ Vector search returned {len(vector_results)} results")
        return vector_results
    
    except Exception as e:
        logger.error(f"✗ Vector search error: {e}")
        return []


# ============================================================================
# HYBRID SEARCH - KEYWORD SEARCH (BM25)
# ============================================================================

def search_keyword_bm25(query: str, documents: List[str], limit: int = 3) -> List[tuple]:
    """
    Sparse keyword search using BM25 algorithm
    Returns: [(id, score, data), ...]
    """
    try:
        from rank_bm25 import BM25Okapi
        
        logger.info(f"🔍 BM25 keyword search for: {query}")
        
        if not documents:
            logger.warning("⚠ No documents provided for BM25 search")
            return []
        
        # Tokenize documents
        tokenized_corpus = [doc.lower().split() for doc in documents]
        tokenized_query = query.lower().split()
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top results
        scored_docs = [(idx, score) for idx, score in enumerate(doc_scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        keyword_results = []
        for idx, score in scored_docs[:limit]:
            if score > 0:
                keyword_results.append((
                    f"bm25_{idx}",
                    float(score),
                    {"title": f"Document {idx}", "content": documents[idx]}
                ))
        
        logger.info(f"✅ BM25 search returned {len(keyword_results)} results")
        return keyword_results
    
    except Exception as e:
        logger.error(f"✗ BM25 search error: {e}")
        return []


# ============================================================================
# HYBRID SEARCH - GRAPH TRAVERSAL
# ============================================================================

async def search_graph_neo4j(user_id: str, limit: int = 3) -> List[tuple]:
    """
    Graph traversal in Neo4j to find user's episodic memories (1-2 hops)
    Uses async driver
    Returns: [(id, relevance_score, data), ...]
    """
    if not clients.neo4j_driver:
        logger.warning("⚠ Neo4j not available for graph search")
        return []
    
    try:
        logger.info(f"🔍 Neo4j graph traversal for user: {user_id}")
        
        async with clients.neo4j_driver.session() as session:
            # Cypher query: Find entities related to user through relationships
            query = """
            MATCH (s:Entity)-[r:RELATIONSHIP {user_id: $user_id}]->(o:Entity)
            WITH DISTINCT o, r, s
            ORDER BY r.timestamp DESC
            LIMIT $limit
            RETURN 
                o.name as title,
                r.type as relation_type,
                s.name as subject_name,
                r.timestamp as timestamp,
                0.7 as relevance_score
            """
            
            results = await session.run(query, user_id=user_id, limit=limit)
            
            graph_results = []
            async for record in results:
                result_data = record.data()
                graph_results.append((
                    f"graph_{len(graph_results)}",
                    float(result_data.get("relevance_score", 0.5)),
                    {
                        "title": result_data.get("title", "Unknown Memory"),
                        "content": f"{result_data.get('subject_name', 'You')} [{result_data.get('relation_type', 'RELATED_TO')}] {result_data.get('title', 'something')}",
                        "timestamp": str(result_data.get("timestamp", "")),
                        "relation_type": result_data.get("relation_type", "")
                    }
                ))
            
            logger.info(f"✅ Graph traversal returned {len(graph_results)} results")
            return graph_results
    
    except Exception as e:
        logger.error(f"✗ Neo4j graph search error: {e}")
        return []


# ============================================================================
# RESPONSE GENERATION - CONTEXT FORMATTING
# ============================================================================

def format_context_for_generation(context_payload: Dict[str, Any]) -> str:
    """Format retrieved context for LLM injection"""
    try:
        context_text = ""
        
        if context_payload.get("retrieved_documents"):
            context_text += "📚 Retrieved Information:\n"
            for idx, doc in enumerate(context_payload["retrieved_documents"][:3], 1):
                context_text += f"{idx}. {doc.get('title', 'Unknown')}\n"
                context_text += f"   {doc.get('content', '')}\n"
                context_text += f"   [Relevance: {doc.get('rrf_score', 0):.2f}]\n"
        
        return context_text
    except Exception as e:
        logger.error(f"Error formatting context: {e}")
        return ""


# ============================================================================
# RESPONSE GENERATION - LLM CALL
# ============================================================================

def generate_empathetic_response(user_id: str, query: str, context_payload: Dict[str, Any], tone: str = "empathetic") -> tuple:
    """
    Generate empathetic, personalized response using LLM with enriched context
    
    Includes hallucination checking to ensure responses are:
    1. Grounded in retrieved context (not hallucinated)
    2. Free of generic AI fallback phrases
    3. Authentic and personalized
    
    Returns:
        (response_text, confidence, citations)
    """
    if not clients.hf_client:
        logger.error("✗ HF Client not available for response generation")
        raise RuntimeError("HF Client not initialized for response generation")
    
    try:
        # Format context
        formatted_context = format_context_for_generation(context_payload)
        
        # Build system prompt based on tone
        tone_instructions = {
            "empathetic": "Respond with deep empathy and emotional understanding. Validate the user's feelings.",
            "supportive": "Provide supportive, encouraging guidance while maintaining professionalism.",
            "analytical": "Provide thoughtful analysis with clear reasoning and practical insights.",
            "motivational": "Inspire and motivate while providing concrete, actionable advice."
        }
        
        system_prompt = f"""You are a compassionate mental health support assistant.
{tone_instructions.get(tone, tone_instructions['empathetic'])}

Personalization Guidelines:
- Use the user's context to understand their situation
- Reference retrieved information when relevant
- Provide specific, actionable advice
- Maintain professional boundaries
- Foster hope and agency
- Be authentic and avoid generic phrases

{f'Retrieved Context Information:' if formatted_context else ''}
{formatted_context}

Recent Conversation History:
{context_payload.get('recent_conversation', 'No recent history.')}

Respond concisely but thoroughly. Be genuine and helpful. NEVER start responses with "As an AI..." or generic phrases."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        logger.info(f"Generating {tone} response for user {user_id}")
        
        # Call LLM
        response = clients.hf_client.chat_completion(
            messages=messages,
            model=clients.MODEL_NAME,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        if not response or not response.choices:
            logger.error("✗ LLM returned empty response")
            raise ValueError("LLM returned empty response")
        
        response_text = response.choices[0].message.content
        
        if not response_text:
            logger.error("✗ LLM response content is empty")
            raise ValueError("LLM response content is empty")
        
        # ============================================================
        # PHASE 8: HALLUCINATION CHECK & REPAIR
        # ============================================================
        logger.info("🔍 Running hallucination validation on LLM output...")
        response_text, final_confidence = validate_and_repair_response(
            response_text,
            context_payload,
            query,
            tone
        )
        
        # Extract citations from context (with safe type handling)
        citations = []
        try:
            if context_payload.get("retrieved_documents"):
                for doc in context_payload["retrieved_documents"][:3]:
                    sources = doc.get("sources", {})
                    # Handle sources being dict or other types
                    if isinstance(sources, dict):
                        sources_str = ",".join(sources.keys())
                    else:
                        sources_str = str(sources)
                    
                    citations.append({
                        "title": str(doc.get("title", "")),
                        "sources": sources_str,
                        "score": str(round(float(doc.get("rrf_score", 0)), 3))
                    })
        except Exception as cite_err:
            logger.error(f"Error extracting citations: {cite_err}")
            # Continue without citations
        
        # Use validation confidence if grounding check was performed, else estimate
        confidence = final_confidence
        
        logger.info(f"✓ Generated response with {confidence:.2f} confidence (hallucination-validated)")
        return response_text, confidence, citations
        
    except Exception as e:
        logger.error(f"✗ Response generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# RESPONSE GENERATION - NEO4J STORAGE
# ============================================================================

async def store_response_neo4j(user_id: str, query: str, response_text: str, tone: str, confidence: float) -> str:
    """Store generated response in Neo4j with metadata using async driver"""
    if not clients.neo4j_driver:
        logger.warning("⚠ Neo4j not available for response storage")
        return ""
    
    try:
        response_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with clients.neo4j_driver.session() as session:
            cypher = """
            MERGE (u:User {id: $user_id})
            MERGE (g:GeneratedResponse {
                id: $response_id,
                timestamp: datetime($timestamp)
            })
            MERGE (u)-[r:RECEIVED_RESPONSE {
                user_query: $user_query,
                tone: $tone,
                confidence: $confidence,
                created_at: datetime($timestamp)
            }]->(g)
            SET g.query = $user_query
            SET g.response = $response_text
            SET g.tone = $tone
            SET g.confidence = $confidence
            SET g.timestamp = datetime($timestamp)
            RETURN g.id as response_id
            """
            
            result = await session.run(
                cypher,
                user_id=user_id,
                response_id=response_id,
                user_query=query,
                response_text=response_text,
                tone=tone,
                confidence=confidence,
                timestamp=timestamp
            )
            
            logger.info(f"✓ Stored response {response_id} for user {user_id}")
            return response_id
            
    except Exception as e:
        logger.error(f"✗ Response storage error: {e}")
        return ""
