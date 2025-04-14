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
    "colors": [
        # Color names
        "green", "red", "blue", "yellow", "purple", "orange", "pink", "brown", "black", "white",
        "cyan", "magenta", "turquoise", "indigo", "violet", "burgundy", "crimson", "teal", 
        # Color terminology
        "hue", "shade", "tint", "saturation", "brightness", "luminosity", "chroma", "palette",
        "primary color", "secondary color", "tertiary color", "complementary color", "color wheel",
        "pastel", "neon", "neutral", "color theory", "color psychology", "color scheme",
        # Color systems
        "rgb", "cmyk", "hsl", "hsv", "pantone", "hex color", "color code", "color model",
        # Art and design related
        "paint color", "pigment", "dye", "colorant", "color matching", "color mixing",
        "color chart", "color gradient", "color spectrum", "color family", "color swatch"
    ],
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
    Filters out fixed categories that aren't sufficiently relevant to the content.
    """
    try:
        features = extract_features(text)
        word_freq = features["word_freq"]
        tokens = features["tokens"]
        original_text = text  # Keep original for pattern matching
        
        # Calculate category scores
        category_scores = _calculate_category_scores(word_freq, tokens, original_text)
        
        # Filter out fixed categories with low scores
        # This prevents categories like "biology" from being assigned when they aren't truly relevant
        filtered_category_scores = {}
        for category, score in category_scores.items():
            # Add a higher threshold for certain categories to prevent them from appearing 
            # when they're not truly relevant
            threshold = 2.0  # Default threshold
            
            # Specific thresholds for commonly over-assigned categories
            if category in ["biology", "colors", "general"]:
                threshold = 3.0
            
            if score >= threshold:
                filtered_category_scores[category] = score
        
        # If we removed too many categories, restore the original scores
        # This ensures we always have at least some categories to work with
        if len(filtered_category_scores) < 2:
            # Sort categories by score and take only those with meaningful scores
            sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            meaningful_categories = [(cat, score) for cat, score in sorted_scores if score > 1.0]
            
            # Just use the top 3 categories if we need more
            if len(meaningful_categories) > 1:
                filtered_category_scores = {cat: score for cat, score in meaningful_categories[:3]}
            else:
                # If everything was filtered, use the original scores for top categories
                filtered_category_scores = {cat: score for cat, score in sorted_scores[:3]}
        
        # Get the top categories from filtered scores
        top_categories = sorted(filtered_category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format and return the result
        if top_categories:
            result = {
                "primary_category": top_categories[0][0],
                "primary_score": top_categories[0][1],
                "top_categories": [(cat, score) for cat, score in top_categories[:3]],
                "all_scores": filtered_category_scores  # Use the filtered scores
            }
            
            # Add specific search terms based on top category
            result["search_terms"] = generate_search_terms(result["primary_category"], text)
            
            return result
        else:
            # Fallback if no categories were found
            return {
                "primary_category": "general",
                "primary_score": 1.0,
                "top_categories": [("general", 1.0), ("academic", 0.5), ("information", 0.3)],
                "all_scores": {"general": 1.0, "academic": 0.5, "information": 0.3},
                "search_terms": []
            }
            
    except Exception as e:
        logger.error(f"Error in classify_text: {str(e)}")
        # Default to a general category if classification fails
        return {
            "primary_category": "general",
            "primary_score": 1.0,
            "top_categories": [("general", 1.0), ("academic", 0.5), ("information", 0.3)],
            "all_scores": {"general": 1.0, "academic": 0.5, "information": 0.3},
            "search_terms": []
        }

def classify_source_text(source_text):
    """
    Classify a source text to determine if it matches the input text's domain.
    Uses the same algorithm as classify_text but optimized for source comparison.
    Also applies filtering to remove irrelevant fixed categories.
    """
    try:
        features = extract_features(source_text)
        word_freq = features["word_freq"]
        tokens = features["tokens"]
        
        # Get initial category scores
        category_scores = _calculate_category_scores(word_freq, tokens, source_text)
        
        # Apply similar filtering logic as in classify_text
        filtered_category_scores = {}
        for category, score in category_scores.items():
            # Add a higher threshold for certain categories to prevent them from appearing 
            # when they're not truly relevant
            threshold = 2.0  # Default threshold
            
            # Specific thresholds for commonly over-assigned categories
            if category in ["biology", "colors", "general"]:
                threshold = 3.0
            
            if score >= threshold:
                filtered_category_scores[category] = score
        
        # If we removed too many categories, restore some of the original scores
        if len(filtered_category_scores) < 2:
            # Sort categories by score and take only those with meaningful scores
            sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            meaningful_categories = [(cat, score) for cat, score in sorted_scores if score > 1.0]
            
            # Use the top 3 categories if we need more
            if len(meaningful_categories) > 1:
                filtered_category_scores = {cat: score for cat, score in meaningful_categories[:3]}
            else:
                # If everything was filtered, use the original scores for top categories
                filtered_category_scores = {cat: score for cat, score in sorted_scores[:3]}
        
        return filtered_category_scores
    except Exception as e:
        logger.error(f"Error in classify_source_text: {str(e)}")
        # Return a generic classification for error cases
        return {"general": 1.0, "academic": 0.5, "information": 0.3}

def is_source_relevant(input_categories, source_categories, threshold=0.7, source_url=None):
    """
    Determine if a source is relevant based on category matching.
    
    Args:
        input_categories: Dictionary of categories and scores for the input text
        source_categories: Dictionary of categories and scores for the source text
        threshold: Minimum relevance score to consider the source relevant
        source_url: Optional URL of the source, used for additional filtering
        
    Returns:
        Tuple of (is_relevant, relevance_score) where is_relevant is a boolean and
        relevance_score is a float from 0-2.0 indicating the strength of the match
    """
    # Get top three categories from input text
    top_input_categories = sorted(input_categories.items(), key=lambda x: x[1], reverse=True)[:3]
    top_input_category_names = [cat[0] for cat in top_input_categories]
    
    # Check if source's top categories match input's top categories
    source_top_categories = sorted(source_categories.items(), key=lambda x: x[1], reverse=True)[:3]
    source_top_category_names = [cat[0] for cat in source_top_categories]
    
    # Calculate relevance score
    relevance_score = 0.0
    
    # --- Category match scoring ---
    
    # Primary category match (highest weight)
    if source_top_category_names and top_input_category_names and source_top_category_names[0] == top_input_category_names[0]:
        relevance_score += 0.8
    elif not source_top_category_names or not top_input_category_names:
        # If we don't have categories, this is a problem
        logger.warning(f"Missing categories - Input: {top_input_category_names}, Source: {source_top_category_names}")
        return False, 0.0
    
    # --- Domain-specific filters ---
    
    # Special case for "Shades of green" and other color pages - explicitly reject them
    # This handles the issue with the example text about green beetles matching color pages
    if "colors" in source_top_category_names[:2] or any("color" in cat.lower() for cat in source_top_category_names[:2]):
        if "biology" in top_input_category_names:
            logger.info(f"Explicitly rejecting color-related source when input is about biology")
            return False, 0.0
            
    # Special detection for URL patterns that indicate color pages when the text is about biology
    if "biology" in top_input_category_names and source_url and (
        "shades_of_" in source_url.lower() or 
        "/color" in source_url.lower() or 
        "/colours" in source_url.lower() or
        "rgb_" in source_url.lower()
    ):
        logger.info(f"Explicitly rejecting URL with color-related patterns: {source_url}")
        return False, 0.0
    
    # --- Calculate overlapping categories ---
    
    # If any of the source's top categories match input's top categories
    common_categories = set(source_top_category_names) & set(top_input_category_names)
    if common_categories:
        # More overlap = higher score
        relevance_score += 0.2 * len(common_categories)
    
    # --- Special handling for specific domains ---
    
    # Special boosting for historical content
    if "history" in top_input_category_names:
        # Boost Wikipedia articles about history
        if source_url and "wikipedia.org" in source_url.lower() and "history" in source_top_category_names:
            relevance_score += 0.4
            
        # Boost historiography-related sources significantly
        if source_url and (
            "historiography" in source_url.lower() or 
            "historical_writing" in source_url.lower() or
            "history_of_" in source_url.lower()
        ):
            relevance_score += 0.5
        
        # Academic sources are valuable for historical content
        if "education" in source_top_category_names or "literature" in source_top_category_names:
            relevance_score += 0.3
    
    # --- Filtering out mismatches ---
    
    # If the source doesn't have the same primary category as the input, 
    # it needs strong secondary matches to be considered relevant
    if source_top_category_names[0] != top_input_category_names[0]:
        # Need at least one matching category in their top categories
        if not common_categories:
            logger.debug(f"Source primary category doesn't match and no common categories")
            return False, 0.0
    
    # Special handling for "general" category - don't allow very generic sources
    if source_top_category_names[0] == "general" and top_input_category_names[0] != "general":
        logger.debug(f"Rejecting generic source for specific input category")
        return False, 0.0
    
    # --- URL-based boosting ---
    
    # Boost Wikipedia sources (generally more reliable)
    if source_url and "wikipedia.org" in source_url.lower():
        relevance_score += 0.2
        
        # Extra boost for exact topic match in Wikipedia URL 
        if top_input_category_names[0] in source_url.lower():
            relevance_score += 0.3
    
    # Boost educational and academic sources
    if source_url and (".edu" in source_url.lower() or "academic" in source_url.lower()):
        relevance_score += 0.2
        
    logger.debug(f"Source relevance score: {relevance_score}, Input categories: {top_input_category_names}, "
                f"Source categories: {source_top_category_names}")
    
    # Return both the boolean result and the score for ranking purposes
    return relevance_score >= threshold, relevance_score

def _calculate_category_scores(word_freq, tokens, original_text):
    """Internal function to calculate category scores for a text"""
    try:
        # Initialize category scores
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
            r'arthropod',
            r'oviduct',
            r'testes',
            r'reproductive',
            r'abdomen',
            r'thorax',
            r'antenna'
        ]
        
        for pattern in biology_patterns:
            if re.search(pattern, original_text, re.IGNORECASE):
                # If we find a strong biology pattern, give it a big boost
                category_scores["biology"] = category_scores.get("biology", 0) + 5
                
        # Color-specific patterns for detecting color-centric documents
        color_patterns = [
            r'shade[s]? of \w+',  # Shades of green/blue/etc
            r'color wheel',
            r'rgb',
            r'cmyk',
            r'hex code',
            r'primary color',
            r'secondary color',
            r'color theory',
            r'color psychology',
            r'color scheme',
            r'rgb values',
            r'color model',
            r'pantone',
            r'color family'
        ]
        
        # Safely check for color patterns with proper error handling
        if original_text is not None and isinstance(original_text, str):
            for pattern in color_patterns:
                try:
                    if re.search(pattern, original_text, re.IGNORECASE):
                        # If we find a strong color pattern, give it a big boost
                        category_scores["colors"] = category_scores.get("colors", 0) + 5
                except (TypeError, re.error) as e:
                    # Log error but continue processing
                    print(f"Error in regex search for pattern '{pattern}': {e}")
                    continue
                
        # Special case: if we detect insect anatomy terminology BUT also color terminology,
        # strongly favor biology over colors for the green beetle example
        if original_text is not None and isinstance(original_text, str):
            try:
                if "beetle" in original_text.lower() and any(term in original_text.lower() for term in ["abdomen", "thorax", "oviduct", "testes", "reproductive"]):
                    try:
                        if re.search(r'green|yellow|color', original_text, re.IGNORECASE):
                            # This is likely about beetle biology that mentions color, not about color itself
                            category_scores["biology"] = category_scores.get("biology", 0) + 8
                            # Decrease any color category score
                            if "colors" in category_scores:
                                category_scores["colors"] = max(0, category_scores["colors"] - 5)
                    except (TypeError, re.error) as e:
                        print(f"Error in regex search for color terms: {e}")
            except Exception as e:
                print(f"Error in beetle biology detection: {e}")
        
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
        if original_text is not None and isinstance(original_text, str):
            try:
                latin_name_pattern = r'\b[A-Z][a-z]+ [a-z]+\b'
                if re.search(latin_name_pattern, original_text):
                    # Latin names strongly suggest biology texts
                    category_scores["biology"] = max(category_scores.get("biology", 0) * 1.5, 5)
            except (TypeError, re.error) as e:
                print(f"Error in Latin name pattern search: {e}")
        
        # Get the top categories
        top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top 3 categories and their scores
        return category_scores
        
    except Exception as e:
        logger.error(f"Error in _calculate_category_scores: {str(e)}")
        # Return generic categories if there's an error
        return {"general": 1.0, "academic": 0.5}

# Dictionary of important domain-specific terms for each category
DOMAIN_SPECIFIC_TERMS = {
    "biology": [
        # Organism types
        "beetle", "insect", "bug", "arthropod", "species", "specimen", "animal", "organism",
        "plant", "bacteria", "fungus", "virus", "microbe", "parasite", "genus", "taxon",
        # Anatomical parts
        "abdomen", "thorax", "head", "antenna", "wing", "leg", "segment", "exoskeleton",
        "ovipositor", "mandible", "eye", "elytra", "pronotum", "testes", "oviduct",
        "chromosome", "cell", "tissue", "membrane", "organelle", "nucleus", "mitochondria",
        # Life stages
        "larva", "pupa", "nymph", "instar", "juvenile", "imago", "adult", "egg",
        "embryo", "fetus", "seed", "germination", "seedling", "sprout", "growth",
        # Biological processes
        "metamorphosis", "reproduction", "mating", "feeding", "development", "evolution",
        "molting", "diapause", "reproductive", "copulation", "parasitizing", "photosynthesis",
        "respiration", "digestion", "excretion", "circulation", "adaptation", "mutation",
        # Physical characteristics
        "coloration", "pattern", "size", "length", "width", "color", "shape", "weight"
    ],
    
    "technology": [
        # Computing hardware
        "processor", "cpu", "gpu", "ram", "memory", "hard disk", "ssd", "motherboard",
        "server", "network", "router", "switch", "hub", "firewall", "bandwidth", "latency",
        # Software
        "algorithm", "code", "program", "software", "database", "operating system", "api",
        "framework", "library", "compiler", "interpreter", "runtime", "debugger", "function",
        # Development
        "development", "programming", "coding", "testing", "debugging", "deployment",
        "version control", "git", "repository", "commit", "merge", "pull request", "agile",
        # Internet & Web
        "internet", "web", "website", "browser", "frontend", "backend", "cloud", "hosting",
        "domain", "url", "html", "css", "javascript", "http", "https", "rest", "graphql",
        # Emerging tech
        "artificial intelligence", "machine learning", "deep learning", "blockchain",
        "cryptocurrency", "bitcoin", "quantum computing", "virtual reality", "augmented reality",
        "robotics", "automation", "iot", "internet of things", "big data", "data science"
    ],
    
    "history": [
        # Time periods
        "ancient", "medieval", "renaissance", "modern", "prehistoric", "century",
        "era", "period", "age", "dynasty", "epoch", "millennium", "decade", "year",
        # Civilizations
        "empire", "kingdom", "civilization", "society", "culture", "nation", "state",
        "republic", "city-state", "tribe", "colony", "settlement", "territory",
        # People & roles
        "king", "queen", "emperor", "pharaoh", "ruler", "monarch", "president",
        "general", "soldier", "warrior", "peasant", "noble", "aristocrat", "slave",
        # Events
        "war", "battle", "conquest", "invasion", "revolution", "uprising", "rebellion",
        "conflict", "siege", "campaign", "treaty", "alliance", "peace", "armistice",
        # Artifacts
        "artifact", "relic", "remains", "ruin", "monument", "temple", "tomb", 
        "manuscript", "document", "archive", "record", "inscription", "hieroglyph"
    ],
    
    "medicine": [
        # Conditions
        "disease", "disorder", "syndrome", "condition", "illness", "infection", "inflammation",
        "cancer", "tumor", "lesion", "injury", "wound", "fracture", "trauma", "pain",
        # Treatment
        "treatment", "therapy", "medication", "drug", "antibiotic", "vaccine", "surgery",
        "procedure", "intervention", "regimen", "dose", "prescription", "protocol", "cure",
        # Anatomy
        "organ", "tissue", "bone", "muscle", "nerve", "vein", "artery", "blood vessel",
        "brain", "heart", "lung", "liver", "kidney", "joint", "tendon", "ligament",
        # Healthcare
        "doctor", "physician", "surgeon", "nurse", "patient", "hospital", "clinic", 
        "diagnosis", "prognosis", "symptom", "sign", "test", "scan", "x-ray", "mri",
        # Processes
        "healing", "recovery", "remission", "relapse", "acute", "chronic", "congenital",
        "pathology", "etiology", "physiology", "immunity", "metabolism", "respiration"
    ],
    
    "agriculture": [
        # Crops
        "crop", "plant", "seed", "grain", "fruit", "vegetable", "cereal", "legume",
        "corn", "wheat", "rice", "soybean", "cotton", "orchard", "vineyard", "harvest",
        # Farming
        "farm", "farmer", "farming", "agriculture", "cultivation", "planting", "growing",
        "harvesting", "irrigation", "fertilization", "sustainable", "organic", "conventional",
        # Soil & Land
        "soil", "land", "field", "plot", "acre", "hectare", "topsoil", "subsoil", 
        "fertility", "nutrient", "mineral", "erosion", "drainage", "conservation", "rotation",
        # Equipment
        "tractor", "plow", "harvester", "combine", "thresher", "drill", "sprayer", 
        "equipment", "machinery", "implement", "tool", "greenhouse", "silo", "barn",
        # Inputs & Management
        "fertilizer", "pesticide", "herbicide", "insecticide", "fungicide", "manure", 
        "compost", "yield", "productivity", "pest", "weed", "disease", "drought", "frost"
    ],
    
    "environment": [
        # Ecosystems
        "ecosystem", "habitat", "biome", "forest", "wetland", "grassland", "desert",
        "tundra", "reef", "rainforest", "savanna", "estuary", "watershed", "biodiversity",
        # Climate
        "climate", "weather", "temperature", "precipitation", "rainfall", "drought",
        "flood", "storm", "hurricane", "typhoon", "cyclone", "global warming", "climate change",
        # Pollution & Impact
        "pollution", "contamination", "emission", "waste", "sewage", "landfill", "toxin",
        "carbon", "greenhouse gas", "co2", "methane", "ozone", "depletion", "deforestation",
        # Conservation
        "conservation", "protection", "preservation", "restoration", "sustainability",
        "renewable", "efficient", "recycling", "biodegradable", "stewardship", "sanctuary",
        # Resources
        "resource", "energy", "water", "air", "soil", "mineral", "fossil fuel", "solar",
        "wind", "hydro", "geothermal", "biomass", "natural gas", "coal", "oil"
    ],
    
    "economics": [
        # Markets
        "market", "economy", "trade", "commerce", "exchange", "supply", "demand", 
        "price", "cost", "value", "money", "currency", "inflation", "deflation", "recession",
        # Finance
        "finance", "investment", "stock", "bond", "share", "securities", "dividend",
        "interest", "loan", "mortgage", "credit", "debt", "asset", "liability", "equity",
        # Business
        "business", "company", "corporation", "firm", "enterprise", "industry", "sector",
        "profit", "revenue", "income", "loss", "expense", "budget", "audit", "acquisition",
        # Employment
        "employment", "unemployment", "job", "labor", "wage", "salary", "compensation",
        "worker", "employee", "employer", "union", "benefit", "pension", "retirement",
        # Economic concepts
        "gdp", "growth", "fiscal", "monetary", "tax", "tariff", "subsidy", "stimulus",
        "austerity", "privatization", "regulation", "deregulation", "globalization", "exports"
    ],
    
    "politics": [
        # Government structures
        "government", "state", "federal", "democracy", "republic", "monarchy", "dictatorship",
        "parliament", "congress", "senate", "legislature", "judiciary", "executive", "constitution",
        # Processes
        "election", "vote", "ballot", "campaign", "policy", "legislation", "regulation",
        "law", "bill", "referendum", "amendment", "veto", "debate", "approval", "ratification",
        # Roles
        "president", "prime minister", "senator", "representative", "legislator", "congressman",
        "judge", "justice", "minister", "secretary", "cabinet", "ambassador", "diplomat",
        # International relations
        "international", "foreign", "domestic", "diplomacy", "treaty", "alliance", "agreement", 
        "sanctions", "embargo", "negotiation", "summit", "convention", "protocol", "relations",
        # Political concepts
        "liberal", "conservative", "progressive", "radical", "moderate", "left", "right", 
        "center", "sovereignty", "authority", "power", "rights", "citizenship", "constituency"
    ],
    
    "education": [
        # Institutions
        "school", "university", "college", "academy", "institute", "campus", "department",
        "faculty", "administration", "board", "district", "preschool", "elementary", "secondary",
        # Roles
        "student", "teacher", "professor", "instructor", "educator", "faculty", "principal",
        "dean", "superintendent", "counselor", "researcher", "scholar", "undergraduate", "graduate",
        # Learning
        "learning", "study", "education", "knowledge", "curriculum", "course", "class",
        "subject", "discipline", "major", "minor", "seminar", "workshop", "training", "lecture",
        # Assessment
        "test", "exam", "assessment", "evaluation", "grade", "score", "performance", 
        "assignment", "homework", "project", "thesis", "dissertation", "research", "paper",
        # Educational concepts
        "pedagogy", "instruction", "literacy", "numeracy", "skill", "competency", "ability",
        "accreditation", "certification", "diploma", "degree", "credential", "standards", "outcomes"
    ],
    
    "literature": [
        # Forms
        "novel", "poem", "play", "essay", "short story", "novella", "epic", "romance",
        "comedy", "tragedy", "drama", "satire", "fiction", "nonfiction", "memoir", "biography",
        # Elements
        "plot", "character", "setting", "theme", "conflict", "resolution", "climax",
        "protagonist", "antagonist", "narrator", "dialogue", "monologue", "description", "scene",
        # Creation
        "author", "writer", "poet", "playwright", "novelist", "critic", "editor", "publisher",
        "writing", "composition", "revision", "editing", "publication", "manuscript", "draft",
        # Analysis
        "analysis", "interpretation", "criticism", "review", "commentary", "explication",
        "symbol", "metaphor", "simile", "imagery", "allusion", "alliteration", "rhyme", "meter",
        # Literary periods
        "classical", "medieval", "renaissance", "romantic", "realist", "modernist", "postmodern",
        "victorian", "enlightenment", "gothic", "baroque", "naturalist", "surrealist", "beat"
    ]
}

# Define a global list of stopwords and non-informative terms to filter out
NON_INFORMATIVE_TERMS = [
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "he", "she",
    "young", "adult", "old", "new", "many", "few", "several", "various", "different",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "some", "such", "other", "another", "more", "most", "all", "any", "each",
    "very", "quite", "rather", "somewhat", "well", "just", "only", "also", "too", "so",
    "during", "before", "after", "while", "since", "until", "within", "without", "upon", "about",
    "above", "below", "under", "over", "between", "among", "along", "around", "through", 
    "their", "our", "your", "my", "his", "her", "its", "which", "what", "who", "whom", "where",
    "like", "as", "than", "then", "when", "how", "why", "because", "therefore", "thus"
]

def extract_domain_entities(text, category):
    """
    Extract domain-specific entities based on the category.
    This function serves as a dispatcher to appropriate specialized entity extractors.
    """
    # Extract common entities that apply to all categories
    common_entities = extract_common_entities(text)
    
    # Extract domain-specific entities based on category
    if category == "biology":
        domain_entities = extract_biological_entities(text)
    elif category == "technology":
        domain_entities = extract_technology_entities(text)
    elif category == "history":
        domain_entities = extract_historical_entities(text)
    elif category == "medicine":
        domain_entities = extract_medical_entities(text)
    elif category == "agriculture":
        domain_entities = extract_agricultural_entities(text)
    elif category == "environment":
        domain_entities = extract_environmental_entities(text)
    elif category == "economics":
        domain_entities = extract_economic_entities(text)
    elif category == "politics":
        domain_entities = extract_political_entities(text)
    elif category == "education":
        domain_entities = extract_educational_entities(text)
    elif category == "literature":
        domain_entities = extract_literary_entities(text)
    else:
        # Default to common entities only for unknown categories
        return common_entities
    
    # Combine the common and domain-specific entities
    all_entities = list(set(common_entities + domain_entities))
    
    # Filter out non-informative terms
    filtered_entities = [entity for entity in all_entities 
                        if entity.lower() not in NON_INFORMATIVE_TERMS
                        and len(entity) > 2  # Exclude very short entities
                        and entity.lower() not in ["the young", "the adult", "the old"]]  # Explicitly exclude common phrases
    
    return filtered_entities

def extract_common_entities(text):
    """
    Extract entities common to all domains: named entities, numeric values,
    dates, and significant noun phrases.
    """
    entities = []
    
    # Extract capitalized named entities
    for match in re.finditer(r'\b[A-Z][a-zA-Z]{2,}\b', text):
        entity = match.group(0)
        if entity.lower() not in ["the", "and", "but", "for", "nor", "yet", "so", "as", "at", "by", "from"]:
            entities.append(entity)
    
    # Extract numerical values with units
    for match in re.finditer(r'\b([0-9]+(?:\.[0-9]+)?)\s*(percent|kg|m|km|ft|mph|tons?|grams?|dollars?)\b', text, re.IGNORECASE):
        entities.append(match.group(0))
    
    # Extract dates
    for match in re.finditer(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]{1,2},?\s+[0-9]{4}\b', text):
        entities.append(match.group(0))
    
    # Extract years
    for match in re.finditer(r'\b(19|20)[0-9]{2}\b', text):
        entities.append(match.group(0))
    
    # Extract meaningful multi-word phrases (noun phrases)
    noun_phrase_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Multi-word proper nouns
        r'\b([a-z]+\s+(?:system|process|project|program|initiative|strategy|framework|concept))\b'  # Common noun phrases
    ]
    
    for pattern in noun_phrase_patterns:
        for match in re.finditer(pattern, text):
            phrase = match.group(0)
            if len(phrase) > 5:  # Reasonable length
                entities.append(phrase)
    
    return entities

def extract_biological_entities(text):
    """
    Extract meaningful biological entities from the text.
    Focus on organism types, anatomical parts, and biological processes.
    """
    entities = []
    
    # Use domain-specific terms
    important_bio_terms = DOMAIN_SPECIFIC_TERMS["biology"]
    
    # Stopwords and non-informative terms to filter out
    non_informative_terms = [
        "the", "a", "an", "this", "that", "these", "those", "it", "they", "he", "she",
        "young", "adult", "old", "new", "many", "few", "several", "various", "different",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "some", "such", "other", "another", "more", "most", "all", "any", "each"
    ]
    
    # Compile patterns to identify important noun phrases
    # Phrases like "adult beetle", "green coloration", "reproductive season"
    noun_phrase_patterns = [
        r'\b(?:the\s+)?([a-z]+(?!\s+(?:is|are|was|were)))\s+(beetle|insect|bug|larva|pupa)\b',  # adult beetle (avoiding "is")
        r'\b(?:the\s+)?(beetle|insect|bug|larva|pupa)\s+(?:is|has|with)\s+([a-z]+)\b',  # beetle is green
        r'\b(?:the\s+)?(abdomen|thorax|head|wing|leg|antenna|ovipositor|testes|oviduct)\b',  # anatomical parts
        r'\b(?:the\s+)?([a-z]+)\s+(abdomen|thorax|head|wing|leg|antenna)\b',  # green abdomen
        r'\b(?:the\s+)?(reproductive|feeding|mating|development)\s+([a-z]+)\b',  # reproductive season
        r'\b([0-9]+(?:\.[0-9]+)?)\s+(mm|cm|millimeters|centimeters)\s+(?:long|wide|in length|in width)\b'  # 6 millimetres long
    ]

    # Extract entities using patterns
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
    
    # Extract scientific classifications
    sci_patterns = [
        r'\b(?:phylum|class|order|family|genus|species)\s+([A-Z][a-z]+)\b',  # Taxonomic classifications
        r'\b([A-Z][a-z]+)(?:\s+[a-z]+)?\s+(?:is|are)\s+(?:a|an)\s+(?:species|genus|family)\b'  # Descriptions
    ]
    
    for pattern in sci_patterns:
        for match in re.finditer(pattern, text):
            # Extract the taxonomic name
            if match.groups():
                entities.append(match.group(1))
    
    return entities

def extract_technology_entities(text):
    """
    Extract technology-specific entities from text.
    Focuses on software, hardware, programming, and digital concepts.
    """
    entities = []
    
    # Use domain-specific terms
    important_tech_terms = DOMAIN_SPECIFIC_TERMS["technology"]
    
    # Define tech-specific patterns
    tech_patterns = [
        r'\b([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+)\b',  # Software versions, domains, file extensions
        r'\b([A-Z][a-zA-Z0-9]*(?:\+\+|#|\.NET))\b',  # Programming languages like C++, C#
        r'\b([a-z]+\.(js|py|java|rb|c|cpp|h|cs|php|html|css|sql))\b',  # Code files
        r'\b(?:uses?|using|with|based on|built with)\s+([A-Za-z][A-Za-z0-9]+)\b',  # Technologies used
        r'\b([a-z]+)\s+(?:framework|library|platform|language|database|api|stack|algorithm)\b',  # Tech components
        r'\b(?:hardware|device|gadget|processor|chip)(?:\s+called|\s+named)?\s+([A-Za-z0-9]+)\b'  # Hardware references
    ]
    
    # Extract tech entities using patterns
    for pattern in tech_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1).lower()
                if entity not in entities and len(entity) > 2:
                    entities.append(entity)
    
    # Extract single-word important tech terms
    words = re.findall(r'\b([a-z][a-z0-9]{2,})\b', text.lower())
    for word in words:
        if word in important_tech_terms and word not in entities:
            entities.append(word)
    
    # Look for measurements and specifications
    spec_patterns = [
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(GB|MB|TB|GHz|MHz|MP|KB)\b',  # Tech measurements
        r'\b([0-9]+[Kk])\s*(?:resolution|pixels?)\b',  # Resolution
        r'\b([0-9]+)\s*(?:bit|core|threads?)\b'  # Computing specifications
    ]
    
    for pattern in spec_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            spec = match.group(0).lower()
            if spec not in entities:
                entities.append(spec)
    
    return entities

def extract_historical_entities(text):
    """
    Extract history-specific entities from text.
    Focuses on time periods, historical figures, events, and locations.
    """
    entities = []
    
    # Use domain-specific terms
    important_hist_terms = DOMAIN_SPECIFIC_TERMS["history"]
    
    # Define history-specific patterns
    hist_patterns = [
        r'\b(?:in|during|after|before|since)\s+(?:the\s+)?([0-9]{1,2}(?:st|nd|rd|th)?\s+century)\b',  # Centuries
        r'\b(?:the\s+)?(ancient|medieval|renaissance|modern|postmodern|early|late|mid|pre|post)\s+(?:period|era|age|times?)\b',  # Time periods
        r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:empire|kingdom|dynasty|republic|civilization|era|period)\b',  # Named historical entities
        r'\b(?:emperor|king|queen|pharaoh|general|ruler)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Rulers
        r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:war|battle|invasion|conquest|revolution|uprising|rebellion)\b',  # Historical events
        r'\b(?:in|at|near|from)\s+([0-9]{3,4})(?:\s+(?:BCE|CE|AD|BC))?\b'  # Years/dates
    ]
    
    # Extract historical entities using patterns
    for pattern in hist_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1)
                if entity.lower() not in ["the"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important historical terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_hist_terms and word not in entities:
            entities.append(word)
    
    # Look for date ranges that often indicate historical periods
    date_patterns = [
        r'\b(?:from|between)?\s*([0-9]{3,4})\s*(?:to|-|–|until)\s*([0-9]{3,4})\b',  # Date ranges like 1914-1918
        r'\b(?:the\s+)?([0-9]{4}s)\b'  # Decades like 1920s
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            date_range = match.group(0)
            if date_range not in entities:
                entities.append(date_range)
    
    return entities

def extract_medical_entities(text):
    """
    Extract medicine-specific entities from text.
    Focuses on conditions, treatments, anatomy, and medical procedures.
    """
    entities = []
    
    # Use domain-specific terms
    important_med_terms = DOMAIN_SPECIFIC_TERMS["medicine"]
    
    # Define medicine-specific patterns
    med_patterns = [
        r'\b(?:diagnosed|suffering|afflicted)\s+(?:with|from)\s+([a-z]+(?:\s+[a-z]+){0,3})\b',  # Diagnosed conditions
        r'\b(?:the\s+)?([a-z]+)\s+(?:disease|syndrome|disorder|condition|infection)\b',  # Named conditions
        r'\b(?:the\s+)?([a-z]+)\s+(?:treatment|therapy|procedure|surgery|medication)\b',  # Treatments
        r'\b(?:administering|administered|prescribed|taking|given)\s+([a-z]+(?:\s+[a-z]+){0,3})\b',  # Medications
        r'\b(?:the\s+)?([a-z]+)\s+(?:symptoms?|signs?)\b',  # Symptoms
        r'\b(?:patients?|doctors?|physicians?|surgeons?)\s+(?:with|who|that)\s+([a-z]+)\b'  # Patient/doctor relations
    ]
    
    # Extract medical entities using patterns
    for pattern in med_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1).lower()
                if entity not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important medical terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_med_terms and word not in entities:
            entities.append(word)
    
    # Look for measurements and dosages
    dosage_patterns = [
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(mg|g|ml|mcg|cc|units?)\b',  # Medical measurements
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:times|doses|tablets|capsules|pills)\s+(?:a|per)\s+(?:day|week|month)\b'  # Dosage frequency
    ]
    
    for pattern in dosage_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            dosage = match.group(0).lower()
            if dosage not in entities:
                entities.append(dosage)
    
    return entities

def extract_agricultural_entities(text):
    """
    Extract agriculture-specific entities from text.
    Focuses on crops, farming practices, soil, and agricultural processes.
    """
    entities = []
    
    # Use domain-specific terms
    important_ag_terms = DOMAIN_SPECIFIC_TERMS["agriculture"]
    
    # Define agriculture-specific patterns
    ag_patterns = [
        r'\b(?:growing|cultivating|planting|harvesting)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # Cultivation activities
        r'\b(?:the\s+)?([a-z]+)\s+(?:crop|seed|cultivar|variety|plant|grain)\b',  # Plant types
        r'\b(?:the\s+)?([a-z]+)\s+(?:soil|field|farm|plantation|orchard|greenhouse)\b',  # Farm locations
        r'\b(?:using|apply|applying|spreading)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:fertilizer|pesticide|herbicide)\b',  # Agricultural inputs
        r'\b(?:the\s+)?([a-z]+)\s+(?:technique|method|approach|system|practice)\s+(?:of|for|in)\s+(?:farming|agriculture|cultivation)\b',  # Farming methods
        r'\b(?:agricultural|farming|crop)\s+([a-z]+(?:\s+[a-z]+){0,2})\b'  # Agricultural concepts
    ]
    
    # Extract agricultural entities using patterns
    for pattern in ag_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1).lower()
                if entity not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important agricultural terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_ag_terms and word not in entities:
            entities.append(word)
    
    # Look for agricultural measurements
    ag_measurement_patterns = [
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:bushels|acres|hectares|tons)\b',  # Farm measurements
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:kg/ha|lbs/acre)\b',  # Yield measurements
        r'\b(?:yield|production)\s+of\s+([0-9]+(?:\.[0-9]+)?)\b'  # Production numbers
    ]
    
    for pattern in ag_measurement_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            measurement = match.group(0).lower()
            if measurement not in entities:
                entities.append(measurement)
    
    return entities

def extract_environmental_entities(text):
    """
    Extract environment-specific entities from text.
    Focuses on ecosystems, climate, pollution, conservation, and resources.
    """
    entities = []
    
    # Use domain-specific terms
    important_env_terms = DOMAIN_SPECIFIC_TERMS["environment"]
    
    # Define environment-specific patterns
    env_patterns = [
        r'\b(?:the\s+)?([a-z]+)\s+(?:ecosystem|habitat|biome|environment|forest|wetland)\b',  # Ecosystem types
        r'\b(?:the\s+)?([a-z]+)\s+(?:species|wildlife|fauna|flora|conservation|protection)\b',  # Conservation elements
        r'\b(?:effects?|impacts?|consequences?)\s+of\s+([a-z]+(?:\s+[a-z]+){0,3})\b',  # Environmental effects
        r'\b(?:climate|environmental)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # Climate factors
        r'\b(?:the\s+)?([a-z]+)\s+(?:pollution|contamination|waste|emissions?)\b',  # Pollution types
        r'\b(?:renewable|sustainable|conservation)\s+([a-z]+(?:\s+[a-z]+){0,2})\b'  # Sustainability concepts
    ]
    
    # Extract environmental entities using patterns
    for pattern in env_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1).lower()
                if entity not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important environmental terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_env_terms and word not in entities:
            entities.append(word)
    
    # Look for environmental measurements
    env_measurement_patterns = [
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:ppm|ppb|µg/m3|mg/l)\b',  # Environmental measurements
        r'\b([0-9]+(?:\.[0-9]+)?)\s*degrees?\s+(?:Celsius|Fahrenheit|centigrade)\b',  # Temperature
        r'\b(?:increased|decreased|reduced|elevated)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)\b'  # Change measurements
    ]
    
    for pattern in env_measurement_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            measurement = match.group(0).lower()
            if measurement not in entities:
                entities.append(measurement)
    
    return entities

def extract_economic_entities(text):
    """
    Extract economics-specific entities from text.
    Focuses on markets, finance, business, employment, and economic concepts.
    """
    entities = []
    
    # Use domain-specific terms
    important_econ_terms = DOMAIN_SPECIFIC_TERMS["economics"]
    
    # Define economics-specific patterns
    econ_patterns = [
        r'\b(?:the\s+)?([a-z]+)\s+(?:market|economy|industry|sector|index)\b',  # Economic sectors
        r'\b(?:the\s+)?([a-z]+)\s+(?:stock|bond|security|investment|fund|asset)\b',  # Financial instruments
        r'\b(?:the\s+)?([a-z]+)\s+(?:company|corporation|firm|enterprise|business)\b',  # Business entities
        r'\b(?:economic|fiscal|monetary|financial)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # Economic concepts
        r'\b(?:increased|decreased|grew|fell|rose|dropped)\s+(?:by|to)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)\b',  # Financial changes
        r'\b(?:the\s+)?([a-z]+)\s+(?:tax|tariff|subsidy|regulation|deregulation|privatization)\b'  # Policy instruments
    ]
    
    # Extract economic entities using patterns
    for pattern in econ_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1).lower()
                if entity not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important economic terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_econ_terms and word not in entities:
            entities.append(word)
    
    # Look for economic figures and statistics
    econ_stat_patterns = [
        r'\b(?:USD|EUR|GBP|JPY|CNY)?\s*\$?\s*([0-9]+(?:[,.][0-9]+)*)\s*(?:million|billion|trillion)?\b',  # Monetary values
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)\s+(?:growth|increase|decrease|decline|interest|inflation|unemployment)\b',  # Economic rates
        r'\b(?:GDP|revenue|profit|loss|debt|deficit)\s+of\s+(?:USD|EUR|GBP|JPY|CNY)?\s*\$?\s*([0-9]+(?:[,.][0-9]+)*)\b'  # Economic figures
    ]
    
    for pattern in econ_stat_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            stat = match.group(0).lower()
            if stat not in entities:
                entities.append(stat)
    
    return entities

def extract_political_entities(text):
    """
    Extract politics-specific entities from text.
    Focuses on government structures, processes, roles, international relations, and political concepts.
    """
    entities = []
    
    # Use domain-specific terms
    important_pol_terms = DOMAIN_SPECIFIC_TERMS["politics"]
    
    # Define politics-specific patterns
    pol_patterns = [
        r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:government|administration|party|regime)\b',  # Named political entities
        r'\b(?:the\s+)?([a-z]+)\s+(?:election|vote|referendum|campaign|policy|legislation)\b',  # Political processes
        r'\b(?:president|prime minister|senator|representative|minister|secretary)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Political figures
        r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:treaty|agreement|accord|resolution|declaration)\b',  # International agreements
        r'\b(?:political|governmental|diplomatic|legislative|regulatory)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # Political concepts
        r'\b(?:the\s+)?([a-z]+)\s+(?:policy|bill|law|regulation|reform|amendment)\b'  # Policy instruments
    ]
    
    # Extract political entities using patterns
    for pattern in pol_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1)
                # Preserve case for proper nouns, lowercase for concepts
                if not entity[0].isupper():
                    entity = entity.lower()
                if entity.lower() not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important political terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_pol_terms and word not in entities:
            entities.append(word)
    
    # Look for political statistics
    pol_stat_patterns = [
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)\s+(?:of the vote|approval|support|opposition)\b',  # Voting/approval percentages
        r'\b([0-9]+)\s+(?:seats?|votes|representatives|delegates|members|constituencies)\b',  # Electoral figures
        r'\b(?:won|lost|secured|obtained)\s+(?:by a margin of|with)\s+([0-9]+(?:\.[0-9]+)?)\b'  # Electoral margins
    ]
    
    for pattern in pol_stat_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            stat = match.group(0).lower()
            if stat not in entities:
                entities.append(stat)
    
    return entities

def extract_educational_entities(text):
    """
    Extract education-specific entities from text.
    Focuses on institutions, roles, learning, assessment, and educational concepts.
    """
    entities = []
    
    # Use domain-specific terms
    important_edu_terms = DOMAIN_SPECIFIC_TERMS["education"]
    
    # Define education-specific patterns
    edu_patterns = [
        r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:University|College|School|Institute|Academy)\b',  # Educational institutions
        r'\b(?:the\s+)?([a-z]+)\s+(?:education|curriculum|program|course|degree|major|minor)\b',  # Educational offerings
        r'\b(?:professor|teacher|instructor|student|faculty|dean)\s+(?:of|at|in)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # Academic roles
        r'\b(?:study|research|thesis|dissertation|project)\s+(?:on|about|regarding)\s+([a-z]+(?:\s+[a-z]+){0,3})\b',  # Academic works
        r'\b(?:the\s+)?([a-z]+)\s+(?:exam|test|assessment|assignment|grade|score|evaluation)\b',  # Assessment types
        r'\b(?:educational|academic|pedagogical|instructional)\s+([a-z]+(?:\s+[a-z]+){0,2})\b'  # Educational concepts
    ]
    
    # Extract educational entities using patterns
    for pattern in edu_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1)
                # Preserve case for proper nouns, lowercase for concepts
                if not entity[0].isupper():
                    entity = entity.lower()
                if entity.lower() not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important educational terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_edu_terms and word not in entities:
            entities.append(word)
    
    # Look for educational statistics
    edu_stat_patterns = [
        r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)\s+(?:pass|fail|graduation|retention|attendance|enrollment)\b',  # Educational percentages
        r'\b([0-9]+)\s+(?:students|credits|courses|classes|hours|years)\b',  # Educational quantities
        r'\b(?:grade\s+point\s+average|GPA)\s+of\s+([0-9]+(?:\.[0-9]+)?)\b'  # GPA measurements
    ]
    
    for pattern in edu_stat_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            stat = match.group(0).lower()
            if stat not in entities:
                entities.append(stat)
    
    return entities

def extract_literary_entities(text):
    """
    Extract literature-specific entities from text.
    Focuses on forms, elements, creation, analysis, and literary periods.
    """
    entities = []
    
    # Use domain-specific terms
    important_lit_terms = DOMAIN_SPECIFIC_TERMS["literature"]
    
    # Define literature-specific patterns
    lit_patterns = [
        r'\b(?:the\s+novel|poem|play|story|book|work|text)\s+(?:titled|called|named|entitled)\s+"?([A-Za-z][A-Za-z\s]+)"?\b',  # Titled works
        r'\b(?:the\s+)?([a-z]+)\s+(?:novel|poem|play|story|essay|memoir|biography|autobiography)\b',  # Literary forms
        r'\b(?:author|writer|poet|playwright|novelist)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Literary figures
        r'\b(?:the\s+)?([a-z]+)\s+(?:character|protagonist|antagonist|narrator|setting|theme|plot|conflict)\b',  # Literary elements
        r'\b(?:literary|poetic|narrative|stylistic|rhetorical)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # Literary concepts
        r'\b(?:the\s+)?([a-z]+)\s+(?:period|movement|tradition|style|genre|school)\b'  # Literary classifications
    ]
    
    # Extract literary entities using patterns
    for pattern in lit_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract the matched entity
            if match.groups():
                entity = match.group(1)
                # Preserve case for titles and proper nouns, lowercase for concepts
                if not entity[0].isupper():
                    entity = entity.lower()
                if entity.lower() not in ["the", "a", "an"] and entity not in entities:
                    entities.append(entity)
    
    # Extract single-word important literary terms
    words = re.findall(r'\b([a-z]{3,})\b', text.lower())
    for word in words:
        if word in important_lit_terms and word not in entities:
            entities.append(word)
    
    # Look for publication information
    pub_patterns = [
        r'\bpublished\s+(?:in|by)\s+([0-9]{4}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Publication info
        r'\b(?:first|second|third|fourth|fifth|latest|final)\s+(?:edition|volume|installment|chapter)\b',  # Edition info
        r'\b(?:written|authored|composed|created)\s+(?:in|during)\s+(?:the\s+)?([0-9]{4}|[a-z]+\s+(?:period|century|era))\b'  # Creation timeframe
    ]
    
    for pattern in pub_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.groups():
                pub_info = match.group(1)
                if pub_info not in entities:
                    entities.append(pub_info)
    
    return entities

def extract_key_phrases(text):
    """
    Extract key phrases using POS tagging to identify linguistically meaningful structures.
    Extracts all terms in VERB, NOUN, ADVERB, ADJECTIVE categories without stopwords,
    and forms search terms as NOUN PHRASES according to grammar rules.
    
    Features:
    - Handles complex multi-word noun phrases with nesting
    - Supports recursive noun phrase structures
    - Captures compound nouns, noun-noun relationships
    - Properly extracts complete phrases like "internal sex organs"
    """
    # Clean the text first - remove punctuation 
    clean_text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Get stopwords to filter out
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and tag parts of speech
    tokens = word_tokenize(clean_text)
    tagged = nltk.pos_tag(tokens)
    
    # Store all content words (VERB, NOUN, ADVERB, ADJECTIVE without stopwords)
    content_words = []
    for word, tag in tagged:
        # Check if word is a content word (VERB, NOUN, ADVERB, ADJECTIVE)
        if (tag.startswith('VB') or tag.startswith('NN') or tag.startswith('RB') or tag.startswith('JJ')):
            # Only include if not a stopword
            if word.lower() not in stop_words and len(word) > 2:
                content_words.append((word, tag))
    
    # Store all NOUN PHRASES we extract
    noun_phrases = []
    
    # IMPROVED GRAMMAR: Complex, recursive noun phrase extraction
    # This implementation builds complete noun phrases with proper multi-word structure
    
    def extract_complex_np(start_idx):
        """
        Recursively extract a complete noun phrase starting at the given index.
        Returns (phrase_tokens, end_idx, has_noun) tuple.
        """
        i = start_idx
        phrase_tokens = []
        has_noun = False
        
        # Find determiner (optional) - like "the", "a", "an", "this"
        if i < len(tagged) and tagged[i][1] in ['DT', 'PRP$']:
            # Skip the determiner in the final phrase
            i += 1
        
        # Extract complex adjective phrases (adj + optional adverbs)
        adj_start = i
        while i < len(tagged):
            # Adverbs modifying adjectives (e.g., "very large")
            if tagged[i][1].startswith('RB'):
                if tagged[i][0].lower() not in stop_words:
                    phrase_tokens.append(tagged[i][0])
                i += 1
            # Adjectives
            elif tagged[i][1].startswith('JJ'):
                if tagged[i][0].lower() not in stop_words:
                    phrase_tokens.append(tagged[i][0])
                i += 1
            else:
                break
        
        # Extract the noun head and any compound nouns
        noun_start = i
        while i < len(tagged):
            # Nouns (singular, plural, proper)
            if tagged[i][1].startswith('NN'):
                has_noun = True
                if tagged[i][0].lower() not in stop_words:
                    phrase_tokens.append(tagged[i][0])
                i += 1
                
                # Look ahead for compound nouns or noun sequences
                # This ensures we capture things like "sex organs" as a complete unit
                compound_i = i
                compound_tokens = []
                
                while compound_i < len(tagged) and tagged[compound_i][1].startswith('NN'):
                    if tagged[compound_i][0].lower() not in stop_words:
                        compound_tokens.append(tagged[compound_i][0])
                    compound_i += 1
                
                # If found compound nouns, add them and update position
                if compound_tokens:
                    phrase_tokens.extend(compound_tokens)
                    i = compound_i
            else:
                break
        
        # Look for prepositional phrase attachments (recursively)
        # This handles phrases like "anatomy of birds" or "structure of the cell membrane"
        if i < len(tagged) and tagged[i][1] == 'IN':
            prep = tagged[i][0]
            i += 1
            
            # Skip determiners after preposition
            if i < len(tagged) and tagged[i][1] in ['DT', 'PRP$']:
                i += 1
            
            # Recursively extract the noun phrase after the preposition
            pp_tokens, new_i, pp_has_noun = extract_complex_np(i)
            
            # Only attach prep phrase if it contains a noun
            if pp_has_noun and pp_tokens:
                phrase_tokens.append(prep)
                phrase_tokens.extend(pp_tokens)
                i = new_i
        
        return phrase_tokens, i, has_noun
    
    # Extract complex noun phrases from the text
    i = 0
    while i < len(tagged):
        # Try to extract a complex noun phrase starting at position i
        np_tokens, new_i, has_noun = extract_complex_np(i)
        
        # If we found a valid noun phrase, add it
        if has_noun and np_tokens:
            phrase = " ".join(np_tokens)
            if phrase not in noun_phrases and len(phrase) > 2:
                noun_phrases.append(phrase)
        
        # Ensure we always make progress
        if new_i > i:
            i = new_i
        else:
            i += 1
    
    # Extract verb phrases (verb + noun combinations)
    verb_phrases = []
    i = 0
    while i < len(tagged):
        # Find verbs
        if tagged[i][1].startswith('VB'):
            verb = tagged[i][0]
            # Skip auxiliary verbs
            if verb.lower() not in ['is', 'are', 'was', 'were', 'be', 'been', 'being',
                                  'have', 'has', 'had', 'do', 'does', 'did']:
                # Look ahead for object nouns
                j = i + 1
                # Skip determiners
                if j < len(tagged) and tagged[j][1] in ['DT', 'PRP$']:
                    j += 1
                # Find any adjectives
                adj_tokens = []
                while j < len(tagged) and tagged[j][1].startswith('JJ'):
                    if tagged[j][0].lower() not in stop_words:
                        adj_tokens.append(tagged[j][0])
                    j += 1
                # Find nouns
                noun_tokens = []
                while j < len(tagged) and tagged[j][1].startswith('NN'):
                    if tagged[j][0].lower() not in stop_words:
                        noun_tokens.append(tagged[j][0])
                    j += 1
                
                # If we found a noun, create a verb phrase
                if noun_tokens and verb.lower() not in stop_words:
                    vp = [verb] + adj_tokens + noun_tokens
                    phrase = " ".join(vp)
                    if phrase not in verb_phrases and len(phrase) > 2:
                        verb_phrases.append(phrase)
            i += 1
        else:
            i += 1
    
    # Extract adverb phrases (combinations with verbs)
    adverb_phrases = []
    i = 0
    while i < len(tagged):
        # Find adverbs
        if tagged[i][1].startswith('RB') and tagged[i][0].lower() not in stop_words:
            adverb = tagged[i][0]
            # Look ahead for verbs
            j = i + 1
            if j < len(tagged) and tagged[j][1].startswith('VB') and tagged[j][0].lower() not in stop_words:
                phrase = f"{adverb} {tagged[j][0]}"
                if phrase not in adverb_phrases and len(phrase) > 2:
                    adverb_phrases.append(phrase)
            i += 1
        else:
            i += 1
    
    # Extract noun-adjective combinations
    adj_noun_phrases = []
    for i in range(len(tagged) - 1):
        if (tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN') and
            tagged[i][0].lower() not in stop_words and tagged[i+1][0].lower() not in stop_words):
            phrase = f"{tagged[i][0]} {tagged[i+1][0]}"
            if phrase not in adj_noun_phrases and len(phrase) > 2:
                adj_noun_phrases.append(phrase)
    
    # Look for domain-specific patterns
    pattern_phrases = []
    patterns = [
        # Measurements
        r'([0-9]+(?:\.[0-9]+)?)\s+(mm|cm|millimeters|centimeters)\s+(?:long|wide|in length|in width)',
        # Colors
        r'(green|yellow|brown|black|red|orange|blue|purple|white|grey|gray)\s+(?:in\s+)?colou?r(?:ation)?',
        # Physical characteristics
        r'([a-z]+)\s+appearance',
        # Biological terms (specific for beetle texts)
        r'(?:exoskeleton|abdomen|thorax|antenna|elytra|wings?|segments?)',
        # Life stages
        r'(?:larva|pupa|adult|nymph|egg|instar)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            phrase = match.group(0).lower()
            if phrase not in pattern_phrases and len(phrase) > 2:
                pattern_phrases.append(phrase)
    
    # Combine all types of phrases, prioritizing multi-word phrases
    all_phrases = []
    
    # Add multi-word phrases first
    for phrase_list in [noun_phrases, verb_phrases, adverb_phrases, adj_noun_phrases, pattern_phrases]:
        for phrase in phrase_list:
            if len(phrase.split()) > 1 and phrase not in all_phrases:
                all_phrases.append(phrase)
    
    # Add important single words if we don't have enough phrases
    if len(all_phrases) < 5:
        important_single_words = []
        # Extract single NOUN, VERB, ADJ, ADV that aren't already part of phrases
        for word, tag in content_words:
            if word.lower() not in stop_words and len(word) > 3:
                # Check if this word isn't already part of a phrase
                if not any(word.lower() in phrase.lower() for phrase in all_phrases):
                    important_single_words.append(word)
        
        # Add most important single words
        for word in important_single_words[:5]:
            if len(all_phrases) < 10:
                all_phrases.append(word)
    
    # Filter out any remaining phrases that are too short or lack substance
    final_phrases = []
    for phrase in all_phrases:
        words = phrase.split()
        non_stop_words = [word for word in words if word.lower() not in stop_words]
        if non_stop_words and len(phrase) > 2:
            final_phrases.append(phrase)
    
    return final_phrases

def generate_search_terms(category, text):
    """
    Generate targeted search terms based on the identified category and text content.
    This now includes all terms in VERB, NOUN, ADVERB, ADJECTIVE categories without stopwords,
    and forms search terms as NOUN PHRASES according to grammar rules.
    The complete list of noun phrases from the input will be included as search terms.
    The text classification result is also added as a search term.
    """
    search_terms = []
    
    # First, extract ALL key phrases using our comprehensive linguistic analysis
    key_phrases = extract_key_phrases(text)
    logger.info(f"Extracted {len(key_phrases)} key phrases using POS tagging")
    
    # Safety check - problematic terms to filter out
    problematic_terms = ["the young", "the adult", "young", "adult", "the", "a", "an"]
    
    # 1. ADD THE CLASSIFICATION CATEGORY FIRST
    # Always include the category as a search term
    if category != "general":
        search_terms.append(category)
    
    # 2. LOOK FOR SCIENTIFIC NAMES (for biology texts)
    scientific_names = []
    latin_name_matches = re.findall(r'\b([A-Z][a-z]+)\s+([a-z]+)\b', text)
    for genus, species in latin_name_matches:
        scientific_names.append(f"{genus} {species}")
    
    if scientific_names:
        for name in scientific_names[:2]:  # Top 2 scientific names
            if name not in search_terms:
                search_terms.append(name)
    
    # 3. ADD NOUN PHRASES FROM THE TEXT
    # Add all extracted noun phrases, filtering out problematic ones
    valid_phrases = []
    for phrase in key_phrases:
        # Skip phrases that are too short
        if len(phrase) < 3:
            continue
        
        # Skip phrases that match problematic terms exactly
        if phrase.lower() in [p.lower() for p in problematic_terms]:
            continue
        
        # Skip phrases that start with problematic prefixes
        if any(phrase.lower().startswith(p.lower() + " ") for p in ["the", "a", "an"]):
            continue
        
        # Add the filtered phrase
        valid_phrases.append(phrase)
    
    # Add valid phrases as search terms
    for phrase in valid_phrases:
        if phrase not in search_terms:
            search_terms.append(phrase)
    
    # 4. DOMAIN-SPECIFIC COMBINATIONS
    # Extract domain-specific entities
    domain_entities = extract_domain_entities(text, category)
    
    # Filter out problematic domain entities
    filtered_domain_entities = []
    for entity in domain_entities:
        if (entity.lower() not in problematic_terms and 
            not any(entity.lower().startswith(term) for term in ["the ", "a ", "an "])):
            filtered_domain_entities.append(entity)
    
    # If we have domain entities, create combinations
    if filtered_domain_entities:
        # Get primary entity
        primary_entity = get_primary_entity(filtered_domain_entities, category)
        
        if primary_entity:
            # Combine primary entity with category (e.g., "beetle biology")
            combined_term = f"{primary_entity} {category}"
            if combined_term not in search_terms:
                search_terms.append(combined_term)
    
    # 5. ADD WIKIPEDIA SEARCH TERMS
    # For biology or if we detect Latin names, add Wikipedia search
    if category == "biology" or scientific_names:
        if "beetle" in text.lower() and "beetle biology" not in search_terms:
            search_terms.append("beetle biology")
        
        if scientific_names and f"{scientific_names[0]} wikipedia" not in search_terms:
            search_terms.append(f"{scientific_names[0]} wikipedia")
    
    # 6. FALLBACK: If we somehow have no terms, add generic ones for the category
    if not search_terms:
        if category == "biology":
            if "beetle" in text.lower():
                search_terms = ["beetle biology", "insect anatomy"]
            else:
                search_terms = ["biology", "species classification"]
        else:
            search_terms = [category, f"{category} research"]
    
    # 7. FINAL FILTERING: Remove any remaining problematic terms
    filtered_search_terms = []
    for term in search_terms:
        # Skip terms that match problematic terms exactly
        if term.lower() in [p.lower() for p in problematic_terms]:
            continue
            
        # Skip terms that start with problematic prefixes
        if any(term.lower().startswith(p.lower() + " ") for p in ["the", "a", "an"]):
            continue
            
        # The term passed all filters
        filtered_search_terms.append(term)
    
    # If we filtered too aggressively and have no terms, add a safe fallback
    if not filtered_search_terms:
        filtered_search_terms = [category]
    
    # 8. Remove duplicates while preserving order
    unique_terms = []
    for term in filtered_search_terms:
        term_lower = term.lower()
        if not any(term_lower == t.lower() for t in unique_terms):
            unique_terms.append(term)
    
    # Increased limit from 5 to 15 search terms for more comprehensive search
    return unique_terms[:15]

def get_primary_entity(entities, category):
    """
    Identify the primary entity from the list based on the category.
    Returns the most important entity that should be the focus of searches.
    """
    if not entities:
        return None
    
    # Different categories have different priority entity types
    if category == "biology":
        # For biology, prioritize organism types
        organism_types = ["beetle", "insect", "arthropod", "bug", "species", 
                         "plant", "animal", "organism", "bacteria"]
        for organism in organism_types:
            if organism in entities:
                return organism
    
    elif category == "technology":
        # For technology, prioritize platforms, languages, or systems
        tech_types = ["algorithm", "software", "application", "system", "platform", 
                     "language", "framework", "database", "network"]
        for tech in tech_types:
            if tech in entities:
                return tech
    
    elif category == "history":
        # For history, prioritize civilizations, events, or periods
        history_types = ["empire", "civilization", "war", "revolution", "dynasty", 
                        "kingdom", "period", "era", "century"]
        for hist in history_types:
            if hist in entities:
                return hist
    
    elif category == "medicine":
        # For medicine, prioritize conditions or treatments
        med_types = ["disease", "syndrome", "disorder", "condition", "treatment", 
                    "therapy", "procedure", "medication", "drug"]
        for med in med_types:
            if med in entities:
                return med
    
    elif category == "agriculture":
        # For agriculture, prioritize crops or farming methods
        ag_types = ["crop", "plant", "seed", "farm", "farming", "cultivation", 
                   "harvest", "soil", "irrigation"]
        for ag in ag_types:
            if ag in entities:
                return ag
    
    elif category == "environment":
        # For environment, prioritize ecosystems or environmental issues
        env_types = ["ecosystem", "climate", "pollution", "conservation", "habitat", 
                    "species", "biodiversity", "warming", "sustainability"]
        for env in env_types:
            if env in entities:
                return env
    
    elif category == "economics":
        # For economics, prioritize economic concepts or markets
        econ_types = ["market", "economy", "industry", "trade", "investment", 
                     "business", "company", "finance", "banking"]
        for econ in econ_types:
            if econ in entities:
                return econ
    
    elif category == "politics":
        # For politics, prioritize government structures or processes
        pol_types = ["government", "policy", "election", "law", "legislation", 
                    "democracy", "party", "president", "parliament"]
        for pol in pol_types:
            if pol in entities:
                return pol
    
    elif category == "education":
        # For education, prioritize institutional or learning concepts
        edu_types = ["school", "university", "education", "learning", "teaching", 
                    "curriculum", "course", "student", "study"]
        for edu in edu_types:
            if edu in entities:
                return edu
    
    elif category == "literature":
        # For literature, prioritize literary forms or elements
        lit_types = ["novel", "book", "poem", "story", "character", "author", 
                    "writer", "narrative", "plot", "theme"]
        for lit in lit_types:
            if lit in entities:
                return lit
    
    # If no priority entity found, return the longest entity (likely most specific)
    return max(entities, key=len) if entities else None

def get_secondary_entities(entities, primary_entity, category):
    """
    Get secondary entities that complement the primary entity.
    These are used to create more specific search terms.
    """
    if not entities:
        return []
    
    # Remove primary entity from the list
    filtered_entities = [e for e in entities if e != primary_entity]
    
    # Different categories have different secondary entity priorities
    if category == "biology":
        # For biology, prioritize anatomical parts, life stages, or behavior
        bio_secondaries = ["abdomen", "thorax", "head", "wing", "antenna", "reproductive", 
                          "larva", "pupa", "adult", "egg", "feeding", "mating", "development"]
        return [e for e in filtered_entities if e in bio_secondaries][:3]
    
    elif category == "technology":
        # For technology, prioritize technical aspects or components
        tech_secondaries = ["architecture", "interface", "protocol", "module", "function", 
                           "data", "security", "performance", "version", "feature"]
        return [e for e in filtered_entities if e in tech_secondaries][:3]
    
    # Similar logic for other categories...
    # Return longest entities if no specific secondary entities found
    return sorted(filtered_entities, key=len, reverse=True)[:3]

def get_domain_concepts(category):
    """
    Return domain-specific concept terms that can be used to create
    meaningful search combinations.
    """
    concepts = {
        "biology": ["taxonomy", "classification", "ecology", "evolution", "behavior"],
        "technology": ["development", "implementation", "architecture", "standard", "specification"],
        "history": ["historical", "ancient", "medieval", "colonial", "revolution"],
        "medicine": ["treatment", "diagnosis", "symptom", "therapy", "clinical"],
        "agriculture": ["cultivation", "production", "sustainable", "organic", "crop"],
        "environment": ["conservation", "sustainable", "ecological", "climate", "protection"],
        "economics": ["financial", "monetary", "fiscal", "economic", "market"],
        "politics": ["policy", "governance", "legislation", "diplomatic", "electoral"],
        "education": ["learning", "teaching", "academic", "curriculum", "educational"],
        "literature": ["literary", "narrative", "critical", "poetic", "rhetorical"]
    }
    
    return concepts.get(category, [])