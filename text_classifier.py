"""
Text classification module for improving plagiarism detection.
This module aims to categorize input text before searching to narrow down results.
"""

import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Create logger
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Dictionary of categories with their associated keywords
CATEGORY_KEYWORDS = {
    "biology": [
        "species", "organism", "biology", "cell", "dna", "animal", "plant", "genus", 
        "ecosystem", "habitat", "microbiology", "evolution", "taxonomy", "bacteria", 
        "virus", "chromosome", "gene", "ecology", "conservation", "biodiversity",
        "specimen", "wildlife", "classification", "genetic", "organism", "ecology",
        "insect", "beetle", "larvae", "pest", "biological", "entomology", "arthropod",
        "exoskeleton", "invertebrate", "thorax", "abdomen", "metamorphosis", "pupa",
        "antennae", "lepidoptera", "coleoptera", "diptera", "hymenoptera", "hemiptera",
        "ovipositor", "mandible", "molt", "instar", "chrysalis", "nymph", "imago",
        "arthropod", "leaf", "defoliation", "host plant", "egg", "wing", "consume"
    ],
    "technology": [
        "algorithm", "computer", "software", "hardware", "programming", "code", 
        "database", "network", "server", "technology", "internet", "online", 
        "digital", "security", "privacy", "cyber", "application", "system",
        "device", "interface", "platform", "storage", "cloud", "protocol",
        "mobile", "virtual", "encryption", "bandwidth", "processor", "authentication"
    ],
    "history": [
        "history", "century", "ancient", "civilization", "empire", "king", "queen", 
        "revolution", "war", "era", "period", "dynasty", "archaeology", "historical", 
        "medieval", "renaissance", "artifact", "colonization", "prehistoric",
        "archive", "chronicle", "document", "heritage", "legacy", "monument",
        "kingdom", "treaty", "conquest", "reign", "timeline"
    ],
    "medicine": [
        "disease", "treatment", "symptom", "patient", "clinical", "medical", 
        "diagnosis", "therapy", "pharmaceutical", "medicine", "healthcare", "drug", 
        "surgery", "prescription", "infection", "physician", "hospital", "nursing",
        "pathology", "vaccine", "immunity", "antibody", "syndrome", "recovery",
        "dose", "chronic", "acute", "prevention", "emergency", "remedy"
    ],
    "agriculture": [
        "farming", "crop", "soil", "fertilizer", "harvest", "irrigation", "agriculture", 
        "pesticide", "livestock", "farm", "cultivation", "organic", "seed", "plantation", 
        "yield", "pasture", "agronomy", "sustainable", "rotation", "greenhouse",
        "horticulture", "gardening", "planting", "weed", "pest", "plant",
        "fruit", "vegetable", "root", "leaf", "botanical"
    ],
    "environment": [
        "climate", "environmental", "ecosystem", "pollution", "conservation", "sustainable", 
        "renewable", "resource", "biodiversity", "habitat", "ecology", "emission", 
        "waste", "carbon", "forest", "wildlife", "endangered", "preservation", "nature",
        "drought", "flood", "erosion", "contamination", "recyclable", "organic",
        "green", "fossil", "energy", "atmosphere", "biosphere"
    ],
    "economics": [
        "economy", "market", "finance", "investment", "trade", "business", "capital", 
        "economic", "inflation", "monetary", "fiscal", "recession", "stock", "currency", 
        "asset", "profit", "commodity", "demand", "supply", "price", "commercial",
        "labor", "industry", "consumer", "production", "wealth", "banking",
        "entrepreneur", "retail", "wholesale"
    ],
    "politics": [
        "government", "policy", "political", "election", "democracy", "vote", "party", 
        "legislation", "regulation", "constitution", "diplomatic", "governance", "law", 
        "politician", "campaign", "referendum", "parliament", "representative", "senate",
        "congress", "court", "judicial", "legislative", "executive", "treaty",
        "sanction", "international", "domestic", "foreign", "administration"
    ],
    "education": [
        "education", "school", "university", "college", "student", "teacher", "professor", 
        "academic", "learning", "curriculum", "classroom", "study", "research", "lecture", 
        "graduate", "undergraduate", "degree", "diploma", "literacy", "educational",
        "course", "program", "faculty", "discipline", "instruction", "pedagogy",
        "teaching", "knowledge", "training", "skill"
    ],
    "literature": [
        "literature", "author", "book", "novel", "poetry", "fiction", "character", 
        "narrative", "literary", "writing", "writer", "publication", "genre", "prose", 
        "drama", "manuscript", "chapter", "essay", "critic", "bibliography",
        "anthology", "metaphor", "imagery", "theme", "plot", "setting",
        "protagonist", "dialogue", "narration", "verse"
    ]
}

def extract_features(text):
    """Extract key features from the text for classification."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Get word frequency
    word_freq = Counter(tokens)
    
    # Extract named entities (capitalized words from original text)
    named_entities = []
    for match in re.finditer(r'\b[A-Z][a-zA-Z]*\b', text):
        named_entities.append(match.group(0).lower())
    
    return {
        "word_freq": word_freq,
        "tokens": tokens,
        "named_entities": named_entities
    }

def classify_text(text):
    """
    Classify the input text based on its content and keywords.
    Returns a dictionary with the top 3 categories and their scores.
    """
    try:
        features = extract_features(text)
        word_freq = features["word_freq"]
        tokens = features["tokens"]
        original_text = text  # Keep original for pattern matching
        
        # Calculate scores for each category
        category_scores = {}
        
        # First check for specific patterns that strongly indicate a particular category
        
        # Biology/species description patterns
        biology_patterns = [
            r'[A-Z][a-z]+ [a-z]+ is a species',
            r'genus [A-Z][a-z]+',
            r'[A-Z][a-z]+ \([A-Z][a-z]+\)',  # Scientific name pattern
            r'larva[el]?',
            r'adult specimens',
            r'feeds on',
            r'host plants?',
            r'life cycle',
            r'pupa[el]?',
            r'insect species',
            r'beetle',
            r'arthropod'
        ]
        
        for pattern in biology_patterns:
            if re.search(pattern, original_text, re.IGNORECASE):
                # If we find a strong biology pattern, give it a big boost
                category_scores["biology"] = category_scores.get("biology", 0) + 5
        
        # Calculate keyword-based scores for all categories
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = category_scores.get(category, 0)  # Get existing score if any
            
            # Score based on keyword matches
            for keyword in keywords:
                if ' ' in keyword:  # Multi-word keyword
                    if keyword in original_text.lower():
                        score += 3  # Multi-word matches are stronger indicators
                elif keyword in tokens:
                    # Get frequency of the keyword
                    freq = word_freq.get(keyword, 0)
                    # Add to score (more frequent keywords count more)
                    score += freq * 2
                elif any(keyword in token for token in tokens):
                    # Partial matches count less
                    score += 0.5
            
            # Normalize by number of keywords to avoid bias towards categories with more keywords
            normalized_score = score / len(keywords) * 10  # Scale from 0-10
            
            category_scores[category] = normalized_score
        
        # Special case for scientific descriptions - check for Latin names
        latin_name_pattern = r'\b[A-Z][a-z]+ [a-z]+\b'
        if re.search(latin_name_pattern, original_text):
            # Latin names strongly suggest biology texts
            category_scores["biology"] = max(category_scores.get("biology", 0) * 1.5, 5)
        
        # Get the top categories
        top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top 3 categories and their scores
        result = {
            "primary_category": top_categories[0][0],
            "primary_score": top_categories[0][1],
            "top_categories": [(cat, score) for cat, score in top_categories[:3]],
            "all_scores": category_scores
        }
        
        # Add specific search terms based on top category
        result["search_terms"] = generate_search_terms(result["primary_category"], text)
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying text: {str(e)}")
        # Return generic classification if there's an error
        return {
            "primary_category": "general",
            "primary_score": 0,
            "top_categories": [("general", 0)],
            "search_terms": []
        }

def generate_search_terms(category, text):
    """Generate targeted search terms based on the identified category and text content."""
    # Extract potential entities (nouns)
    words = text.split()
    entities = []
    scientific_names = []
    
    # Look for scientific names (Latin binomials) - highest priority for biology texts
    latin_name_matches = re.findall(r'\b([A-Z][a-z]+)\s+([a-z]+)\b', text)
    for genus, species in latin_name_matches:
        scientific_names.append(f"{genus} {species}")
    
    # Look for capitalized words that might be important entities
    for word in words:
        if word and word[0].isupper() and len(word) > 3:
            # Remove any punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word not in entities:
                entities.append(clean_word)
    
    # Extract most common words (excluding stopwords)
    features = extract_features(text)
    common_words = features["word_freq"].most_common(5)
    common_terms = [word for word, freq in common_words]
    
    # Generate search queries
    search_terms = []
    
    # Handle specific categories differently
    if category == "biology":
        # For biology, prioritize scientific names (if any)
        for name in scientific_names[:2]:  # Top 2 scientific names
            search_terms.append(name)
            # Add scientific name with category for more precision
            search_terms.append(f"{name} {category}")
            # For encyclopedic content, searching with "wikipedia" is helpful
            search_terms.append(f"{name} wikipedia")
        
        # If no scientific names found, use regular entities with biology-specific context
        if not scientific_names and entities:
            search_terms.append(f"{entities[0]} {category}")
            search_terms.append(f"{entities[0]} species")
            # Add some common biology terms
            if len(entities) >= 2:
                search_terms.append(f"{entities[0]} {entities[1]}")
    else:
        # Standard search term generation for other categories
        if category != "general" and category in CATEGORY_KEYWORDS:
            # Include the most relevant entity with the category
            if entities:
                search_terms.append(f"{entities[0]} {category}")
            
            # Add a search combining common terms with the category
            if common_terms:
                search_terms.append(f"{common_terms[0]} {category}")
        
        # Add entity-based searches
        for entity in entities[:2]:  # Limit to top 2 entities
            search_terms.append(entity)
        
        # Add combination of entities if we have multiple
        if len(entities) >= 2:
            search_terms.append(f"{entities[0]} {entities[1]}")
    
    # Add entity with common word if available and not already added
    if entities and common_terms and len(search_terms) < 5:
        search_terms.append(f"{entities[0]} {common_terms[0]}")
    
    # Remove duplicates and limit to max 5 search terms
    return list(dict.fromkeys(search_terms))[:5]