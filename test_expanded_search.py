#!/usr/bin/env python
"""
Test script for the expanded search term limit.
This script verifies that we're generating up to 15 search terms now.
"""

import logging
from text_classifier import extract_key_phrases, generate_search_terms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Test texts with lots of potential search terms
TEST_CASES = [
    """
    The internal reproductive system of beetles is highly specialized. Male beetles possess 
    testes that produce sperm cells, which are transferred to females during mating. Female 
    beetles have complex internal sex organs including ovaries for egg production, spermatheca 
    for sperm storage, and ovipositors for egg-laying. The structure of these reproductive organs 
    varies significantly across different beetle species, reflecting evolutionary adaptations to 
    diverse ecological niches. Reproductive strategies in beetles range from simple direct fertilization 
    to complex behaviors involving specialized structures for sperm competition. 
    """,
    
    """
    Beetles' exoskeletons provide protection and structural support. Made primarily of chitin, 
    these hardened shells defend against predators and environmental stresses. The exoskeleton 
    is divided into three main sections: the head, thorax, and abdomen. On the head, beetles 
    possess antennae for sensing their environment, compound eyes for vision, and specialized 
    mouthparts for feeding. The thorax bears three pairs of legs for locomotion, while many beetle 
    species also feature two pairs of wings - the hardened front pair (elytra) protecting the 
    membranous hind wings used for flight. Additionally, the respiratory system of beetles includes 
    spiracles on their abdomen, allowing for gas exchange. This combination of features has 
    contributed to beetles becoming the most diverse animal group on Earth.
    """
]

def test_expanded_search_terms():
    """Test the expanded limit of search terms."""
    for i, text in enumerate(TEST_CASES):
        logger.info(f"Test case #{i+1}")
        logger.info("=" * 80)
        logger.info(text.strip())
        logger.info("-" * 80)
        
        # Extract key phrases
        phrases = extract_key_phrases(text)
        
        # Generate search terms with expanded limit
        search_terms = generate_search_terms("biology", text)
        
        # Print extracted phrases
        logger.info(f"Extracted {len(phrases)} key phrases:")
        for j, phrase in enumerate(phrases):
            logger.info(f"  {j+1}. {phrase}")
        
        # Print search terms
        logger.info(f"\nGenerated {len(search_terms)} search terms:")
        for j, term in enumerate(search_terms):
            logger.info(f"  {j+1}. {term}")
            
        logger.info("=" * 80)

if __name__ == "__main__":
    logger.info("Testing expanded search term limit (up to 15 terms)")
    logger.info("=" * 80)
    test_expanded_search_terms()
    logger.info("Testing complete!")