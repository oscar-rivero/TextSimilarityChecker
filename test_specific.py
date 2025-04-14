#!/usr/bin/env python
"""
Special test focused specifically on the "internal sex organs" example.
This script tests how our grammar handles this case consistently.
"""

import logging
from text_classifier import extract_key_phrases, generate_search_terms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Test specifically on variations of the "internal sex organs" phrase
TEST_CASES = [
    "Beetles have internal sex organs for reproduction.",
    "The internal sex organs of beetles are complex.",
    "Scientists study the internal sex organs to understand beetle reproduction.",
    "Insects like beetles develop specialized internal sex organs during metamorphosis.",
    "Comparative studies of internal sex organs across beetle species reveal evolutionary adaptations."
]

def test_specific_phrase():
    """Specifically test the extraction of 'internal sex organs'."""
    for i, text in enumerate(TEST_CASES):
        logger.info(f"Test case #{i+1}: {text}")
        
        # Extract key phrases
        phrases = extract_key_phrases(text)
        
        # Print extracted phrases
        logger.info(f"Extracted phrases:")
        for phrase in phrases:
            logger.info(f"  - {phrase}")
        
        # Also test the search term generation
        search_terms = generate_search_terms("biology", text)
        logger.info(f"Generated search terms:")
        for term in search_terms:
            logger.info(f"  - {term}")
            
        logger.info("-" * 60)

if __name__ == "__main__":
    logger.info("Testing extraction of 'internal sex organs' phrase")
    logger.info("=" * 80)
    test_specific_phrase()
    logger.info("Testing complete!")