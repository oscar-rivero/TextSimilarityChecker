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

# Dictionary of categories with their associated keywords (enhanced with Wikipedia categories)
CATEGORY_KEYWORDS = {
    "biology": [
        # Core biology terms
        "species", "organism", "biology", "cell", "dna", "animal", "plant", "genus", 
        "ecosystem", "habitat", "microbiology", "evolution", "taxonomy", "bacteria", 
        "virus", "chromosome", "gene", "ecology", "conservation", "biodiversity",
        "specimen", "wildlife", "classification", "genetic", "organism", "ecology",
        
        # Insect and entomology specific
        "insect", "beetle", "larvae", "pest", "biological", "entomology", "arthropod",
        "exoskeleton", "invertebrate", "thorax", "abdomen", "metamorphosis", "pupa",
        "antennae", "lepidoptera", "coleoptera", "diptera", "hymenoptera", "hemiptera",
        "ovipositor", "mandible", "molt", "instar", "chrysalis", "nymph", "imago",
        "leaf", "defoliation", "host plant", "egg", "wing", "consume",
        
        # Wikipedia biology categories
        "zoology", "botany", "biota", "flora", "fauna", "mycology", "ornithology", 
        "ichthyology", "herpetology", "mammalogy", "phylogenetics", "molecular biology",
        "systematics", "embryology", "morphology", "comparative anatomy", "physiology",
        "photosynthesis", "respiration", "metabolism", "protein", "enzyme", "organelle",
        "mitochondria", "chloroplast", "ribosome", "natural selection", "adaptation",
        "reproductive", "pollination", "germination", "life cycle", "symbiosis",
        "predator", "prey", "parasite", "host", "biome", "biomass", "population",
        "community ecology", "food web", "food chain", "trophic level", "keystone species"
    ],
    "technology": [
        # Core technology terms
        "algorithm", "computer", "software", "hardware", "programming", "code", 
        "database", "network", "server", "technology", "internet", "online", 
        "digital", "security", "privacy", "cyber", "application", "system",
        "device", "interface", "platform", "storage", "cloud", "protocol",
        "mobile", "virtual", "encryption", "bandwidth", "processor", "authentication",
        
        # Wikipedia technology categories
        "information technology", "computer science", "artificial intelligence", 
        "machine learning", "robotics", "automation", "biotechnology", "nanotechnology", 
        "telecommunications", "electronics", "computing", "engineering", "manufacturing",
        "aerospace", "automotive", "industrial", "materials science", "nuclear technology", 
        "renewable energy", "sustainable technology", "medical technology", "assistive technology",
        "emerging technologies", "data science", "internet of things", "blockchain", 
        "quantum computing", "augmented reality", "virtual reality", "3d printing",
        "big data", "data mining", "cryptography", "devops", "agile", "waterfall",
        "gpu", "cpu", "microcontroller", "firmware", "api", "sdk", "framework"
    ],
    "history": [
        # Core history terms
        "history", "century", "ancient", "civilization", "empire", "king", "queen", 
        "revolution", "war", "era", "period", "dynasty", "archaeology", "historical", 
        "medieval", "renaissance", "artifact", "colonization", "prehistoric",
        "archive", "chronicle", "document", "heritage", "legacy", "monument",
        "kingdom", "treaty", "conquest", "reign", "timeline",
        
        # Wikipedia history categories
        "history by location", "history by period", "history by topic", "chronology", 
        "historiography", "archaeological sites", "antiquity", "classical antiquity", 
        "bronze age", "iron age", "middle ages", "early modern period", "modern history", 
        "contemporary history", "world war", "cold war", "civil war", "military history",
        "political history", "social history", "economic history", "cultural history",
        "intellectual history", "maritime history", "art history", "history of science", 
        "history of technology", "historical figures", "ancient rome", "ancient greece", 
        "ancient egypt", "medieval europe", "byzantine empire", "ottoman empire",
        "holy roman empire", "colonial america", "french revolution", "industrial revolution",
        "ancient civilizations", "historical documents", "primary sources"
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

def extract_biological_entities(text):
    """
    Extract meaningful biological entities from the text.
    Focus on organism types, anatomical parts, and biological processes.
    """
    # Define biological and anatomical terms that would be important in descriptions
    important_bio_terms = [
        # Organism types
        "beetle", "insect", "bug", "arthropod", "species", "specimen", "animal",
        # Anatomical parts
        "abdomen", "thorax", "head", "antenna", "wing", "leg", "segment", "exoskeleton",
        "ovipositor", "mandible", "eye", "elytra", "pronotum", "testes", "oviduct",
        # Life stages
        "larva", "pupa", "nymph", "instar", "juvenile", "imago", "adult", "egg",
        # Biological processes
        "metamorphosis", "reproduction", "mating", "feeding", "development", 
        "molting", "diapause", "reproductive", "copulation", "parasitizing",
        # Physical characteristics
        "coloration", "pattern", "size", "length", "width", "color", "shape"
    ]
    
    # Compile patterns to identify important noun phrases
    # Phrases like "adult beetle", "green coloration", "reproductive season"
    noun_phrase_patterns = [
        r'\b(?:the\s+)?([a-z]+)\s+(beetle|insect|bug|adult|larva|pupa)\b',  # adult beetle
        r'\b(?:the\s+)?(beetle|insect|bug|adult|larva|pupa)\s+(?:is|has|with)\s+([a-z]+)\b',  # beetle is green
        r'\b(?:the\s+)?(abdomen|thorax|head|wing|leg|antenna|ovipositor)\b',  # anatomical parts
        r'\b(?:the\s+)?([a-z]+)\s+(abdomen|thorax|head|wing|leg|antenna)\b',  # green abdomen
        r'\b(?:the\s+)?(reproductive|feeding|mating|development)\s+([a-z]+)\b',  # reproductive season
        r'\b([0-9]+(?:\.[0-9]+)?)\s+(mm|cm|millimeters|centimeters)\s+(?:long|wide|in length|in width)\b'  # 6 millimetres long
    ]

    # Extract entities using patterns
    entities = []
    for pattern in noun_phrase_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get all groups from match
            groups = match.groups()
            # Extract useful phrases
            for group in groups:
                if group and len(group) > 2 and group.lower() not in ["the", "has", "is", "with"]:
                    # Clean and add entity
                    clean_entity = group.lower().strip()
                    if clean_entity not in entities:
                        entities.append(clean_entity)

    # Extract single-word important biological terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_bio_terms and word not in entities:
            entities.append(word)
    
    # Extract noun phrases with specific biological meaning
    if "beetle" in text.lower() or "insect" in text.lower():
        if "adult" in text.lower() and "adult beetle" not in entities and "beetle" not in entities:
            entities.append("beetle")
        if "young" in text.lower() and "larva" not in entities:
            entities.append("larva")
    
    # Look for color descriptions as they're often important in insect identification
    color_pattern = r'\b(?:is|are|appears|colou?red|coloration)\s+([a-z]+)\b'
    color_matches = re.finditer(color_pattern, text, re.IGNORECASE)
    for match in color_matches:
        color = match.group(1).lower()
        if color not in ["is", "are", "the", "and", "or", "very", "quite", "somewhat"]:
            if color not in entities:
                entities.append(color)
    
    return entities

def extract_key_phrases(text):
    """Extract key phrases that might be important for searching."""
    phrases = []
    
    # Look for phrases that describe characteristics
    patterns = [
        r'([0-9]+(?:\.[0-9]+)?)\s+(mm|cm|millimeters|centimeters)\s+(?:long|wide|in length|in width)',  # Size descriptions
        r'(green|yellow|brown|black|red|orange|blue|purple|white|grey|gray)\s+(?:in\s+)?colou?r',  # Color descriptions
        r'([a-z]+)\s+(?:in\s+)?appearance',  # Appearance descriptions
        r'(?:feeds|feeding)\s+on\s+([a-z\s]+)',  # Feeding habits
        r'(?:live|lives|found)\s+(?:in|on)\s+([a-z\s]+)',  # Habitat descriptions
        r'(?:during|in)\s+(?:its|the)\s+([a-z]+\s+[a-z]+)'  # Life cycle phases
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            phrase = match.group(0).lower()
            if phrase and len(phrase) > 5:  # Reasonable length for a phrase
                phrases.append(phrase)
    
    return phrases

def generate_search_terms(category, text):
    """Generate targeted search terms based on the identified category and text content."""
    search_terms = []
    scientific_names = []
    
    # Look for scientific names (Latin binomials) - highest priority for biology texts
    latin_name_matches = re.findall(r'\b([A-Z][a-z]+)\s+([a-z]+)\b', text)
    for genus, species in latin_name_matches:
        scientific_names.append(f"{genus} {species}")
    
    # Extract biological entities for biology texts
    if category == "biology":
        # Get biological entities
        bio_entities = extract_biological_entities(text)
        
        # Prioritize scientific names if found
        if scientific_names:
            for name in scientific_names[:2]:  # Top 2 scientific names
                search_terms.append(name)
                # For encyclopedic content, searching with "wikipedia" is helpful
                search_terms.append(f"{name} wikipedia")
                # Add category for good measure
                if len(search_terms) < 4:
                    search_terms.append(f"{name} {category}")
        
        # Add primary biological entities
        if bio_entities:
            # Prioritize organism type (beetle, insect, etc.) if present
            organism_types = ["beetle", "insect", "arthropod", "bug", "species"]
            primary_entity = None
            for organism in organism_types:
                if organism in bio_entities:
                    primary_entity = organism
                    break
            
            # Add primary entity searches
            if primary_entity:
                # Combine primary entity (e.g., "beetle") with category
                if f"{primary_entity} {category}" not in search_terms:
                    search_terms.append(f"{primary_entity} {category}")
                
                # If we have scientific name and primary entity, combine them
                if scientific_names and f"{scientific_names[0]} {primary_entity}" not in search_terms:
                    search_terms.append(f"{scientific_names[0]} {primary_entity}")
                
                # Add specialized search with Wikipedia for encyclopedic content
                if f"{primary_entity} wikipedia" not in search_terms:
                    search_terms.append(f"{primary_entity} wikipedia")
            
            # Add key anatomical or descriptive terms combined with the primary entity or category
            anatomical_parts = ["abdomen", "thorax", "wing", "antenna", "ovipositor", "testes", "reproductive"]
            for part in anatomical_parts:
                if part in bio_entities and len(search_terms) < 5:
                    if primary_entity:
                        search_terms.append(f"{primary_entity} {part}")
                    else:
                        search_terms.append(f"{part} {category}")
                    break  # Just add one anatomical search term
            
            # Add specific searches for remaining biological entities
            remaining_entities = [e for e in bio_entities if e not in search_terms and not any(e in term for term in search_terms)]
            for entity in remaining_entities[:2]:  # Limit to 2 more entities
                if len(search_terms) < 5:
                    if primary_entity and entity != primary_entity:
                        search_terms.append(f"{primary_entity} {entity}")
                    else:
                        search_terms.append(entity)
        
        # If still not enough search terms, add key phrases
        if len(search_terms) < 5:
            key_phrases = extract_key_phrases(text)
            for phrase in key_phrases[:2]:  # Limit to 2 phrases
                if len(search_terms) < 5 and len(phrase) < 30:  # Reasonable length 
                    search_terms.append(phrase)
    
    else:
        # For non-biology categories, fall back to original logic but with enhancements
        # Extract features for common word identification
        features = extract_features(text)
        common_words = features["word_freq"].most_common(10)  # Get more words to have better choices
        
        # Filter for meaningful words (not just common but potentially meaningless words)
        meaningful_words = [word for word, _ in common_words if len(word) > 3 and not word.startswith('the')]
        
        if meaningful_words:
            # Add category-specific search with most meaningful word
            search_terms.append(f"{meaningful_words[0]} {category}")
            
            # Add the top meaningful word by itself
            search_terms.append(meaningful_words[0])
            
            # Add combination of top meaningful words
            if len(meaningful_words) >= 2:
                search_terms.append(f"{meaningful_words[0]} {meaningful_words[1]}")
            
            # Add more meaningful words if needed
            for word in meaningful_words[2:5]:
                if len(search_terms) < 5:
                    search_terms.append(word)
        
        # Look for capitalized entities as they might be important
        cap_entities = []
        for match in re.finditer(r'\b[A-Z][a-zA-Z]{3,}\b', text):
            entity = match.group(0)
            if entity and entity.lower() not in [term.lower() for term in search_terms]:
                cap_entities.append(entity)
        
        # Add capitalized entities to search terms
        for entity in cap_entities[:2]:
            if len(search_terms) < 5:
                search_terms.append(entity)
    
    # Ensure we have at least some search terms
    if not search_terms and category != "general":
        search_terms.append(category)
    
    # Remove duplicates and limit to max 5 search terms
    # Convert to lowercase for easier comparison
    unique_terms = []
    for term in search_terms:
        term_lower = term.lower()
        if not any(term_lower == t.lower() for t in unique_terms):
            unique_terms.append(term)
    
    return unique_terms[:5]