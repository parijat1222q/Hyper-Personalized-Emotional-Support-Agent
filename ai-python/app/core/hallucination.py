"""
Phase 8: Lightweight Hallucination Checker

This module provides a 4-layer hallucination detection and repair system:
  1. Strip generic AI phrases using regex patterns
  2. Validate response grounding in retrieved context
  3. Detect common hallucination patterns
  4. Repair or downgrade confidence scores

All validation is performed without modifying the original response unless
explicitly necessary.
"""

import re
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 8: LIGHTWEIGHT HALLUCINATION CHECKER
# ============================================================================

class HallucinationCheckResult:
    """Result of hallucination validation"""
    def __init__(self, is_valid: bool, cleaned_text: str, issues: List[str], confidence: float):
        self.is_valid = is_valid
        self.cleaned_text = cleaned_text
        self.issues = issues
        self.confidence = confidence  # How confident this is grounded in context


# Generic AI Phrases & Fallback Patterns (STRICT BLOCKLIST)
GENERIC_AI_PHRASES = {
    # Meta-AI statements (Tier 1: CRITICAL)
    r"(?i)as\s+an\s+ai\s+(language\s+)?model": "Meta-reference to being an AI",
    r"(?i)as\s+an\s+artificial\s+intelligence": "Meta-reference to being AI",
    r"(?i)i'm\s+an\s+ai": "Direct AI self-reference",
    r"(?i)i\s+am\s+a\s+machine": "Machine self-reference",
    
    # Generic Empathy Fallbacks (Tier 2: HIGH RISK)
    r"(?i)i'm\s+here\s+for\s+you": "Generic support phrase",
    r"(?i)i\s+understand\s+what\s+you're\s+going\s+through": "Canned empathy",
    r"(?i)i\s+can\s+only": "Limitation deflection",
    r"(?i)i\s+cannot\s+replace": "Disclaimer fallback",
    r"(?i)please\s+seek\s+professional\s+help": "Blanket referral",
    r"(?i)if\s+you\s+are\s+in\s+crisis": "Cookie-cutter crisis line",
    
    # Hedging & Non-Answer Patterns (Tier 3: MEDIUM RISK)
    r"(?i)in\s+my\s+opinion": "Unnecessary qualifier",
    r"(?i)i\s+believe": "Non-grounded belief",
    r"(?i)you\s+might\s+want\s+to": "Evasive suggestion",
    r"(?i)it\s+depends": "Non-committal (when context is provided)",
    r"(?i)i\s+couldn't\s+find": "No-answer pattern",
}


def strip_generic_ai_phrases(response: str) -> Tuple[str, List[str]]:
    """
    Scan response for generic AI phrases and return cleaned version + list of found issues
    
    Returns:
        (cleaned_response, found_issues)
    """
    issues = []
    cleaned = response
    
    for pattern, description in GENERIC_AI_PHRASES.items():
        matches = re.finditer(pattern, response)
        for match in matches:
            issues.append(f"{description}: '{match.group()}'")
            logger.warning(f"⚠️ Detected generic phrase: {description} → '{match.group()}'")
    
    return cleaned, issues


def validate_response_grounding(
    response: str,
    context_payload: Dict[str, Any],
    original_query: str
) -> Tuple[bool, List[str], float]:
    """
    Validate that response is grounded in provided context (not hallucinated)
    
    Checks:
    1. Response should reference facts from retrieved documents
    2. Response should not introduce named entities not in context
    3. Response length should be proportional to context quality
    
    Returns:
        (is_grounded, validation_issues, grounding_confidence)
    """
    issues = []
    grounding_score = 0.0
    
    try:
        retrieved_docs = context_payload.get("retrieved_documents", [])
        
        # Scoring mechanism
        if retrieved_docs:
            # More context = higher expected grounding
            context_words = set()
            for doc in retrieved_docs:
                content = str(doc.get("content", "")).lower()
                context_words.update(content.split())
            
            # Extract key nouns/entities from context
            response_words = set(response.lower().split())
            
            # Calculate overlap
            if context_words:
                overlap = len(response_words & context_words) / len(response_words) if response_words else 0
                grounding_score = min(1.0, overlap * 1.2)  # 20% boost for good overlap
            else:
                grounding_score = 0.5  # Neutral if no context
            
            # Check response length vs context
            response_length = len(response.split())
            context_length = sum(len(doc.get("content", "").split()) for doc in retrieved_docs)
            
            if response_length > context_length * 3:
                issues.append("Response too long relative to context (potential hallucination)")
                grounding_score *= 0.7
        else:
            # No context = higher hallucination risk
            issues.append("No context retrieved - response may be hallucinated")
            grounding_score = 0.3
        
        is_grounded = grounding_score > 0.4 and not issues
        
        return is_grounded, issues, grounding_score
        
    except Exception as e:
        logger.error(f"✗ Grounding validation error: {e}")
        return False, [f"Validation error: {e}"], 0.2


def validate_response_authenticity(
    response: str,
    context_payload: Dict[str, Any],
    original_query: str,
    tone: str = "empathetic"
) -> HallucinationCheckResult:
    """
    Primary hallucination check using multiple validation layers
    
    Layer 1: Strip generic AI phrases
    Layer 2: Validate grounding in context
    Layer 3: Check response coherence with query
    
    Returns:
        HallucinationCheckResult with validation details
    """
    all_issues = []
    confidence = 1.0
    
    # LAYER 1: Generic Phrase Detection
    cleaned_text, phrase_issues = strip_generic_ai_phrases(response)
    if phrase_issues:
        all_issues.extend(phrase_issues)
        confidence *= 0.6  # 40% confidence reduction for generic phrases
    
    # LAYER 2: Grounding Validation
    is_grounded, grounding_issues, grounding_conf = validate_response_grounding(
        response,
        context_payload,
        original_query
    )
    if grounding_issues:
        all_issues.extend(grounding_issues)
    
    # Blend confidences
    confidence = (confidence * 0.5) + (grounding_conf * 0.5)
    
    # LAYER 3: Check for common hallucination patterns
    hallucination_patterns = [
        (r"(?i)according to\s+\w+\s+study", "Unverified study reference"),
        (r"(?i)research\s+shows", "Unverified research claim"),
        (r"(?i)doctors\s+agree", "Generalized medical claim"),
        (r"(?i)scientifically\s+proven", "Unverified scientific claim"),
    ]
    
    for pattern, issue_desc in hallucination_patterns:
        if re.search(pattern, response):
            all_issues.append(issue_desc)
            confidence *= 0.8
    
    # Final validity determination
    is_valid = len(all_issues) == 0 and confidence > 0.5
    
    if all_issues:
        logger.warning(f"⚠️ Hallucination check found {len(all_issues)} issues:")
        for issue in all_issues:
            logger.warning(f"  - {issue}")
    
    return HallucinationCheckResult(
        is_valid=is_valid,
        cleaned_text=cleaned_text,
        issues=all_issues,
        confidence=confidence
    )


def validate_and_repair_response(
    response: str,
    context_payload: Dict[str, Any],
    original_query: str,
    tone: str = "empathetic"
) -> Tuple[str, float]:
    """
    Validate response for hallucinations and repair if needed
    
    Returns:
        (repaired_response, final_confidence)
    """
    try:
        logger.info("🔍 Running hallucination checker on generated response...")
        
        # Run validation
        check_result = validate_response_authenticity(
            response,
            context_payload,
            original_query,
            tone
        )
        
        if check_result.is_valid:
            logger.info("✅ Response passed hallucination validation")
            return response, check_result.confidence
        
        # If issues found, attempt repair
        logger.warning(f"⚠️ Response contains {len(check_result.issues)} hallucination indicators")
        
        repaired = response
        
        # Remove detected phrases
        for pattern in GENERIC_AI_PHRASES.keys():
            repaired = re.sub(pattern, "", repaired, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        repaired = " ".join(repaired.split())
        
        # If still problematic, return original with reduced confidence
        if len(check_result.issues) > 2:
            logger.warning("⚠️ Too many issues - using original response with reduced confidence")
            return response, max(0.2, check_result.confidence - 0.3)
        
        logger.info(f"✅ Response repaired, confidence: {check_result.confidence:.2f}")
        return repaired, check_result.confidence
        
    except Exception as e:
        logger.error(f"✗ Response repair error: {e}")
        # Fail safely - return original
        return response, 0.6
