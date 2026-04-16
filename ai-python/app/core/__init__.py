"""
OmniMind AI Worker - Core Package

Core utilities including hallucination checking and validation.
"""

from .hallucination import (
    HallucinationCheckResult,
    GENERIC_AI_PHRASES,
    strip_generic_ai_phrases,
    validate_response_grounding,
    validate_response_authenticity,
    validate_and_repair_response,
)

__all__ = [
    "HallucinationCheckResult",
    "GENERIC_AI_PHRASES",
    "strip_generic_ai_phrases",
    "validate_response_grounding",
    "validate_response_authenticity",
    "validate_and_repair_response",
]
