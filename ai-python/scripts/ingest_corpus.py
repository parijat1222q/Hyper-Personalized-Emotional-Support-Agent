#!/usr/bin/env python3
"""
OmniMind Knowledge Base Ingestion Script
Scrapes mental health coping techniques and embeds them into Qdrant vector database
with intelligent fallback for robustness.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Vector embedding & database
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-vector")  
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # ~27MB, runs on CPU
VECTOR_SIZE = 384  # BGE-small output dimension

# Scraping Configuration
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
CHUNK_SIZE = 300  # characters per chunk

# Fallback Clinical Knowledge Base (High-Quality Techniques)
FALLBACK_KNOWLEDGE_BASE = [
    {
        "name": "5-4-3-2-1 Grounding Technique",
        "category": "Grounding",
        "description": "Engage your senses to stay present. Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. This technique helps interrupt anxiety and panic by redirecting focus to the immediate sensory environment.",
        "instructions": "1. Notice 5 things you can see around you\n2. Notice 4 things you can physically touch\n3. Notice 3 things you can hear\n4. Notice 2 things you can smell\n5. Notice 1 thing you can taste\nThis simple sensory awareness technique takes 5-10 minutes and is highly effective for dissociation and anxiety.",
        "effectiveness": "Highly effective for anxiety, panic attacks, and dissociation"
    },
    {
        "name": "Box Breathing Technique",
        "category": "Breathing",
        "description": "A 4-count breathing pattern used by military and athletes to calm the nervous system. Breathe in for 4 counts, hold for 4, exhale for 4, hold for 4. This technique activates the parasympathetic nervous system.",
        "instructions": "1. Breathe in slowly through nose for 4 counts\n2. Hold breath for 4 counts\n3. Exhale slowly through mouth for 4 counts\n4. Hold breath for 4 counts\n5. Repeat 5-10 cycles\nThis works best in a quiet space and takes about 3-5 minutes.",
        "effectiveness": "Excellent for panic attacks, acute stress, and sleep preparation"
    },
    {
        "name": "Progressive Muscle Relaxation",
        "category": "Relaxation",
        "description": "Systematically tense and release muscle groups to reduce physical anxiety symptoms. Start with feet and move upward through the body, tensing for 5 seconds then releasing.",
        "instructions": "1. Find a comfortable position\n2. Starting with toes, tense muscles for 5 seconds\n3. Release and notice the relaxation for 10 seconds\n4. Move to calves, thighs, abdomen, chest, arms, shoulders, neck, face\n5. Complete cycle typically takes 15-20 minutes\nThis technique helps you recognize the difference between tension and relaxation.",
        "effectiveness": "Proven for reducing muscle tension, anxiety, and insomnia"
    },
    {
        "name": "Cognitive Behavioral Therapy (CBT) Thought Record",
        "category": "Cognitive",
        "description": "A structured journaling technique to challenge automatic negative thoughts. Record the situation, thought, feeling, evidence for/against, and rational response.",
        "instructions": "1. Write down the triggering situation\n2. Identify the automatic negative thought\n3. Rate the emotion intensity (0-100)\n4. List evidence supporting the thought\n5. List evidence contradicting the thought\n6. Write a more balanced, realistic thought\n7. Re-rate the emotion intensity\nThis technique helps break cycles of negative thinking patterns.",
        "effectiveness": "Foundation of CBT - highly effective for depression and anxiety"
    },
    {
        "name": "Mindfulness Meditation for Anxiety",
        "category": "Mindfulness",
        "description": "Non-judgmental awareness of present moment without trying to change thoughts. Notice thoughts and feelings as they arise, observe them, let them pass.",
        "instructions": "1. Find a quiet space, sit comfortably\n2. Close eyes or maintain soft gaze\n3. Focus on natural breathing\n4. When mind wanders (normal), gently return to breath\n5. Notice thoughts/emotions but don't engage with them\n6. Practice for 10-20 minutes daily\nStart with 5 minutes if new to meditation.",
        "effectiveness": "Reduces anxiety, improves emotional regulation, increases present-moment awareness"
    },
    {
        "name": "Loving-Kindness Meditation",
        "category": "Mindfulness",
        "description": "Cultivate compassion by directing well-wishes toward yourself and others. Counteracts self-criticism and builds emotional resilience.",
        "instructions": "1. Sit comfortably, close eyes\n2. Silently repeat: 'May I be happy, may I be healthy, may I be safe, may I live with ease'\n3. Extend these wishes to a loved one\n4. Extend to a neutral person\n5. Extend to someone difficult\n6. Extend to all beings\nPractice 15-20 minutes daily for best results.",
        "effectiveness": "Reduces self-criticism, increases emotional connection, helps with social anxiety"
    },
    {
        "name": "Journaling for Emotional Processing",
        "category": "Expressive",
        "description": "Free-form written expression of emotions and thoughts without censorship. Helps externalize internal struggles and gain perspective.",
        "instructions": "1. Set aside 15-20 minutes with pen and paper\n2. Write continuously without stopping\n3. Don't worry about grammar, spelling, or coherence\n4. Express whatever emotions arise - anger, sadness, fear, joy\n5. Don't re-read immediately\n6. Optional: Review after a day for insights\nResearch shows processing trauma through writing accelerates healing.",
        "effectiveness": "Effective for trauma processing, emotional awareness, and reducing rumination"
    },
    {
        "name": "Behavioral Activation for Depression",
        "category": "Behavioral",
        "description": "Combat depression by scheduling and completing meaningful activities. Depression creates avoidance cycles; breaking them through action helps mood improve.",
        "instructions": "1. List activities that usually bring satisfaction\n2. Schedule specific times for these activities\n3. Start small (even 15 minutes counts)\n4. Track mood before and after each activity\n5. Gradually increase complexity and duration\n6. Don't wait to 'feel like it' - action creates mood, not vice versa\nThis is a core depression treatment in clinical practice.",
        "effectiveness": "Highly effective first-line treatment for depression"
    },
    {
        "name": "Exposure Therapy for Anxiety Disorders",
        "category": "Exposure",
        "description": "Gradually face feared situations or thoughts in safe settings to reduce anxiety responses. The brain learns the feared outcome doesn't occur.",
        "instructions": "1. Create a hierarchy of feared situations (least to most anxiety)\n2. Start with lower-anxiety situations\n3. Stay in situation until anxiety naturally decreases (30-60 min)\n4. Repeat until anxiety level drops significantly\n5. Move to next level\n6. Continue hierarchy progression\nAlways approach with therapist support for trauma-related fears.",
        "effectiveness": "Gold standard for anxiety disorders, PTSD, and phobias"
    },
    {
        "name": "Somatic Experiencing Techniques",
        "category": "Somatic",
        "description": "Body-focused awareness to discharge trauma stored in the nervous system. Notice body sensations, tension, temperature, and movement impulses.",
        "instructions": "1. Sit quietly and scan your body for tension\n2. Notice areas of tightness without judgment\n3. Observe your natural breathing rhythm\n4. Notice any urges to move (stretch, shake, sway)\n5. Allow these movements to happen naturally\n6. Don't force expression - let it emerge\n7. Practice for 10-15 minutes\nThis technique is rooted in how trauma is stored in the body.",
        "effectiveness": "Effective for trauma, panic disorder, and chronic anxiety"
    },
    {
        "name": "Dialectical Behavior Therapy (DBT) TIPP Skill",
        "category": "DBT",
        "description": "Quick crisis skill using Temperature, Intense exercise, Paced breathing, and Paired muscle relaxation to regulate extreme emotions.",
        "instructions": "Choose one: (T) Splash face with cold water or hold ice; (I) Do intense exercise for 1-2 min; (P) Breathe in for 5 counts, exhale longer; (P) Tense and release muscles. Use when emotions feel unmanageable and you need immediate relief within minutes.",
        "effectiveness": "Immediate emotion regulation in crisis moments - proven in DBT programs"
    },
    {
        "name": "Acceptance and Commitment Therapy (ACT) Values Work",
        "category": "ACT",
        "description": "Identify personal core values, acknowledge painful thoughts without fighting them, and commit to values-aligned actions despite discomfort.",
        "instructions": "1. Identify 5-7 life areas (relationships, career, health, etc.)\n2. For each area, write what truly matters to you\n3. Rate current alignment with values (0-10)\n4. Identify values-aligned actions you can take\n5. Commit to small steps this week\n6. Accept that discomfort may arise during values pursuit\nThis builds psychological flexibility and meaningful living.",
        "effectiveness": "Effective for depression, anxiety, chronic pain, and improving life meaning"
    },
    {
        "name": "Sleep Hygiene Practices for Insomnia",
        "category": "Sleep",
        "description": "Evidence-based practices to improve sleep quality: consistent sleep schedule, cool dark room, no screens 1 hour before bed, limit caffeine after 2pm.",
        "instructions": "1. Go to bed and wake at same time daily (even weekends)\n2. Keep bedroom cool (65-68°F), dark, and quiet\n3. Remove electronic devices from bedroom\n4. No caffeine after 2pm, no alcohol 3-4 hours before bed\n5. Exercise daily but not within 3 hours of bedtime\n6. If awake 20 min, get up and do calm activity until drowsy\nConsistent practice yields results within 2-4 weeks.",
        "effectiveness": "Foundation of insomnia treatment - first-line intervention"
    },
    {
        "name": "Self-Compassion Breaks for Stress",
        "category": "Self-Compassion",
        "description": "When stressed, pause and offer yourself the same kindness you'd give a friend. Recognize suffering as part of humanity, practice self-kindness.",
        "instructions": "1. Acknowledge: 'This is a moment of suffering. Suffering is part of life.'\n2. Self-kindness: Place hand on heart, say kind words to yourself\n3. Common humanity: 'Others feel this way too. I'm not alone.'\nRe-research shows self-compassion is more effective than self-esteem for resilience.",
        "effectiveness": "Reduces anxiety and depression, increases resilience and emotional flexibility"
    },
    {
        "name": "Cognitive Defusion Techniques",
        "category": "Cognitive",
        "description": "Distance yourself from unhelpful thoughts by observing them without believing or fighting them. Think 'I'm having the thought that...' instead of accepting it as truth.",
        "instructions": "1. Notice the anxious thought\n2. Mentally label it: 'I'm having the thought that X'\n3. Observe it as if watching clouds pass in sky\n4. Don't argue with or suppress the thought\n5. Let it pass naturally while continuing activities\n6. Practice recognizing thought as 'thought' not 'fact'\nThis reduces thought control struggles and anxiety amplification.",
        "effectiveness": "Highly effective for obsessive thoughts, anxiety, and rumination"
    }
]

# Sources for scraping (with fallback)
SCRAPE_SOURCES = [
    {
        "url": "https://www.mentalhealth.gov/basics/mental-health",
        "parser": "generic_html"
    },
    {
        "url": "https://www.mhanational.org/mental-health-information",
        "parser": "generic_html"
    }
]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into overlapping chunks for better embedding context."""
    chunks = []
    sentences = text.split('. ')
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [c for c in chunks if len(c.strip()) > 20]  # Filter tiny chunks


def scrape_mental_health_content() -> List[Dict[str, str]]:
    """
    Attempt to scrape mental health content from trusted sources.
    Returns list of technique documents or empty list on failure.
    """
    logger.info("🔍 Attempting to scrape mental health content...")
    
    scraped_content = []
    
    for source in SCRAPE_SOURCES:
        try:
            logger.info(f"  Scraping: {source['url']}")
            
            headers = {
                "User-Agent": "OmniMind-KB-Ingestion/1.0 (Educational, Mental Health Knowledge Corpus)"
            }
            
            response = requests.get(
                source['url'],
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract main content
            text_content = ""
            for tag in soup.find_all(['p', 'li', 'h2', 'h3']):
                text = tag.get_text(strip=True)
                if len(text) > 20 and len(text) < 1000:
                    text_content += text + " "
            
            if text_content:
                scraped_content.append({
                    "name": f"Content from {source['url'].split('/')[-1]}",
                    "description": text_content[:500],
                    "full_text": text_content,
                    "source": source['url'],
                    "source_type": "scraped"
                })
                logger.info(f"  ✓ Successfully scraped {len(text_content)} characters")
        
        except Exception as e:
            logger.warning(f"  ✗ Failed to scrape {source['url']}: {e}")
            continue
    
    return scraped_content


def prepare_knowledge_documents() -> List[Dict[str, Any]]:
    """
    Prepare knowledge documents: combine scraped content with fallback clinical base.
    This ensures robustness - if scraping fails, we still have quality content.
    """
    logger.info("📚 Preparing knowledge documents...")
    
    documents = []
    
    # Try to scrape first
    scraped = scrape_mental_health_content()
    if scraped:
        logger.info(f"✓ Using {len(scraped)} scraped sources")
        documents.extend(scraped)
    else:
        logger.warning("⚠ Scraping failed, using fallback clinical knowledge base")
    
    # Always include our high-quality fallback techniques
    logger.info(f"✓ Adding {len(FALLBACK_KNOWLEDGE_BASE)} clinical techniques")
    
    for technique in FALLBACK_KNOWLEDGE_BASE:
        doc = {
            "name": technique["name"],
            "description": technique["description"],
            "category": technique.get("category", "General"),
            "instructions": technique.get("instructions", ""),
            "effectiveness": technique.get("effectiveness", ""),
            "source": "OmniMind Clinical Knowledge Base",
            "source_type": "verified_clinical"
        }
        documents.append(doc)
    
    logger.info(f"📦 Total documents prepared: {len(documents)}")
    return documents


def embed_and_prepare_points(
    documents: List[Dict[str, Any]],
    embedding_model: TextEmbedding
) -> List[PointStruct]:
    """
    Embed documents and prepare PointStruct objects for Qdrant upsert.
    """
    logger.info("🧠 Generating embeddings...")
    
    points = []
    
    for idx, doc in enumerate(documents):
        try:
            # Combine text fields for embedding
            text_to_embed = f"{doc.get('name', '')} {doc.get('description', '')} {doc.get('instructions', '')}"
            
            if not text_to_embed.strip():
                logger.warning(f"  ⚠ Skipping document {idx}: empty content")
                continue
            
            # Generate embedding
            embeddings_generator = embedding_model.embed(text_to_embed)
            embedding = list(embeddings_generator)[0]  # Convert generator to list and get first item
            
            # Create unique ID
            point_id = int(hashlib.md5(
                (doc.get('name', '') + str(idx)).encode()
            ).hexdigest()[:16], 16) % (2**31)
            
            # Prepare payload
            payload = {
                "name": doc.get("name", "Unknown"),
                "description": doc.get("description", ""),
                "category": doc.get("category", "General"),
                "instructions": doc.get("instructions", ""),
                "effectiveness": doc.get("effectiveness", ""),
                "source": doc.get("source", ""),
                "source_type": doc.get("source_type", "general"),
                "ingested_at": datetime.utcnow().isoformat(),
                "full_text": text_to_embed
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            points.append(point)
            
            if (idx + 1) % 5 == 0:
                logger.info(f"  ✓ Embedded {idx + 1}/{len(documents)} documents")
        
        except Exception as e:
            logger.error(f"  ✗ Error embedding document {idx}: {e}")
            continue
    
    logger.info(f"✓ Successfully embedded {len(points)} documents")
    return points


def initialize_qdrant_collection(client: QdrantClient) -> bool:
    """Initialize Qdrant collection with proper schema if it doesn't exist."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            logger.info(f"✓ Collection '{COLLECTION_NAME}' already exists")
            return True
        
        logger.info(f"📋 Creating collection '{COLLECTION_NAME}'...")
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        logger.info(f"✓ Collection '{COLLECTION_NAME}' created successfully")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize collection: {e}")
        return False


def ingest_into_qdrant(client: QdrantClient, points: List[PointStruct]) -> bool:
    """Upsert points into Qdrant knowledge base collection."""
    try:
        if not points:
            logger.warning("⚠ No points to ingest")
            return False
        
        logger.info(f"📤 Upserting {len(points)} points into Qdrant...")
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"✓ Successfully upserted {len(points)} points")
        
        # Verify ingestion
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info(f"📊 Collection stats: {collection_info.points_count} total points")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to upsert into Qdrant: {e}")
        return False


def main():
    """Main ingestion pipeline."""
    logger.info("=" * 70)
    logger.info("🚀 OmniMind Knowledge Base Ingestion")
    logger.info("=" * 70)
    
    try:
        # 1. Connect to Qdrant
        logger.info(f"\n📍 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("✓ Connected to Qdrant")
        
        # 2. Initialize collection
        if not initialize_qdrant_collection(qdrant_client):
            logger.error("✗ Failed to initialize Qdrant collection")
            return False
        
        # 3. Prepare knowledge documents
        logger.info("\n📚 Preparing Knowledge Base...")
        documents = prepare_knowledge_documents()
        
        # 4. Initialize embedding model (runs locally on CPU)
        logger.info(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
        logger.info("   ⏳ First load may take 30-60 seconds (downloading model)...")
        embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info(f"✓ Embedding model loaded (vector size: {VECTOR_SIZE})")
        
        # 5. Embed and prepare points
        logger.info("\n🔄 Embedding documents...")
        points = embed_and_prepare_points(documents, embedding_model)
        
        if not points:
            logger.error("✗ No points generated for embedding")
            return False
        
        # 6. Ingest into Qdrant
        logger.info("\n💾 Ingesting into Qdrant...")
        success = ingest_into_qdrant(qdrant_client, points)
        
        logger.info("\n" + "=" * 70)
        if success:
            logger.info("✅ Knowledge Base Ingestion Complete!")
            logger.info(f"   • {len(points)} semantic documents embedded")
            logger.info(f"   • {len(FALLBACK_KNOWLEDGE_BASE)} verified clinical techniques loaded")
            logger.info(f"   • Collection: {COLLECTION_NAME}")
            logger.info(f"   • Vector dimension: {VECTOR_SIZE}")
            logger.info("=" * 70)
            return True
        else:
            logger.error("❌ Ingestion failed")
            return False
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
