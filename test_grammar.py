#!/usr/bin/env python
"""
Test script for the improved grammar implementation in text_classifier.py
This script tests our ability to properly extract complex noun phrases.
"""

import logging
import sys
from text_classifier import extract_key_phrases

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Sample texts with complex noun phrases
SAMPLE_TEXTS = [
    # Basic test with "internal sex organs"
    "The beetles have internal sex organs that are specialized for reproduction.",
    
    # More complex structure with nested noun phrases
    "Beetles possess complex reproductive systems with specialized internal structures for sperm storage.",
    
    # Test with prepositional phrases
    "The anatomy of beetles includes organs of reproduction located within the abdomen.",
    
    # Test with compound nouns and adjectives
    "Large black beetles with hard shiny exoskeletons are common in tropical forests.",
    
    # Test with recursive structure (nested prepositional phrases)
    "The structure of the cell membrane of beetles contains specialized proteins for chemical detection.",
    
    # Test with adjective sequences
    "Shiny metallic green beetles are prized by collectors for their beautiful appearance.",
    
    # Test with adverbs modifying adjectives
    "Extremely large tropical beetles can grow to impressive sizes in the rainforest.",
]

def test_grammar_extraction():
    """Test the improved grammar extraction for noun phrases."""
    for i, text in enumerate(SAMPLE_TEXTS):
        logger.info(f"Test case #{i+1}: {text}")
        
        # Extract key phrases
        phrases = extract_key_phrases(text)
        
        # Print extracted phrases
        logger.info(f"Extracted {len(phrases)} key phrases:")
        for j, phrase in enumerate(phrases):
            logger.info(f"  {j+1}. {phrase}")
        
        logger.info("-" * 60)

if __name__ == "__main__":
    logger.info("Testing improved grammar implementation for key phrase extraction")
    logger.info("=" * 80)
    test_grammar_extraction()
    logger.info("Grammar testing complete!")