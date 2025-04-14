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
    
    # Extract domain-specific entities based on the category
    domain_entities = extract_domain_entities(text, category)
    
    # Safety check - ensure we don't have problematic tokens
    problematic_terms = ["the young", "the adult", "young", "adult", "the"]
    
    # Create a second list of filtered domain entities to be extra safe
    filtered_domain_entities = []
    for entity in domain_entities:
        if entity.lower() not in problematic_terms and not any(entity.lower().startswith(term) for term in ["the ", "a "]):
            filtered_domain_entities.append(entity)
    
    # For any category, if we have scientific names, they're usually high-value search terms
    if scientific_names:
        for name in scientific_names[:2]:  # Top 2 scientific names
            search_terms.append(name)
            # For encyclopedic content, searching with "wikipedia" is helpful
            search_terms.append(f"{name} wikipedia")
            # Add category for good measure
            if len(search_terms) < 4:
                search_terms.append(f"{name} {category}")
    
    # If we have domain-specific entities, use them intelligently
    if filtered_domain_entities:
        # Determine if there's a primary entity type for this domain
        primary_entity = get_primary_entity(filtered_domain_entities, category)
        
        # Be extra sure our primary entity isn't in the problematic list
        if primary_entity and primary_entity.lower() in problematic_terms:
            # Try to find another entity
            filtered_domain_entities = [e for e in filtered_domain_entities if e.lower() not in problematic_terms]
            primary_entity = get_primary_entity(filtered_domain_entities, category) if filtered_domain_entities else None
        
        # Add primary entity searches
        if primary_entity:
            # Combine primary entity with category (e.g., "beetle biology")
            if f"{primary_entity} {category}" not in search_terms:
                search_terms.append(f"{primary_entity} {category}")
            
            # If we have scientific name and primary entity, combine them
            if scientific_names and f"{scientific_names[0]} {primary_entity}" not in search_terms:
                search_terms.append(f"{scientific_names[0]} {primary_entity}")
            
            # Add specialized search with Wikipedia for encyclopedic content
            if f"{primary_entity} wikipedia" not in search_terms:
                search_terms.append(f"{primary_entity} wikipedia")
        
        # Get secondary entities that complement the primary entity
        secondary_entities = get_secondary_entities(filtered_domain_entities, primary_entity, category)
        
        # Add combinations of primary entity with secondary entities
        for sec_entity in secondary_entities[:2]:  # Limit to top 2 secondary entities
            if len(search_terms) < 5:
                if primary_entity and sec_entity != primary_entity:
                    search_terms.append(f"{primary_entity} {sec_entity}")
                else:
                    search_terms.append(sec_entity)
        
        # Add specific search for beetle abdomen or anatomical parts for biology
        if category == "biology" and "beetle" in text.lower():
            anatomical_terms = ["abdomen", "thorax", "head", "wing", "antenna", 
                               "elytra", "leg", "ovipositor", "reproductive"]
            
            for term in anatomical_terms:
                if term in text.lower() and len(search_terms) < 5:
                    search_terms.append(f"beetle {term}")
                    break
    
    # If we don't have enough search terms yet, add key phrases or domain concepts
    if len(search_terms) < 3:
        # For biology, extract key phrases
        if category == "biology":
            key_phrases = extract_key_phrases(text)
            for phrase in key_phrases[:2]:  # Limit to 2 phrases
                if len(search_terms) < 5 and len(phrase) < 30:  # Reasonable length 
                    # Check that phrase doesn't start with problematic terms
                    if not any(phrase.lower().startswith(term) for term in problematic_terms):
                        search_terms.append(phrase)
        else:
            # For non-biology, add domain-specific concept phrases
            domain_concepts = get_domain_concepts(category)
            
            # Extract features for common word identification
            features = extract_features(text)
            common_words = features["word_freq"].most_common(10)
            
            # Filter to ensure we don't use problematic terms
            meaningful_words = [word for word, _ in common_words 
                              if len(word) > 3 
                              and word.lower() not in problematic_terms]
            
            # Combine top meaningful word with domain concept
            if meaningful_words and domain_concepts:
                for concept in domain_concepts[:2]:
                    if len(search_terms) < 5:
                        search_terms.append(f"{meaningful_words[0]} {concept}")
    
    # If we still have no terms after all our checks, use the category
    if not search_terms and category != "general":
        # Add some domain-specific fallback terms
        if category == "biology" and "beetle" in text.lower():
            search_terms.append("beetle biology")
            search_terms.append("insect anatomy")
        else:
            search_terms.append(category)
    
    # Do a final filter to ensure we don't have problematic terms
    filtered_search_terms = []
    for term in search_terms:
        if not any(term.lower() == pt.lower() for pt in problematic_terms) and term.lower() not in problematic_terms:
            filtered_search_terms.append(term)
    
    # If we filtered too aggressively and have no terms, add a safe fallback
    if not filtered_search_terms:
        filtered_search_terms = [category]
    
    # Remove duplicates and limit to max 5 search terms
    # Convert to lowercase for easier comparison
    unique_terms = []
    for term in filtered_search_terms:
        term_lower = term.lower()
        if not any(term_lower == t.lower() for t in unique_terms):
            unique_terms.append(term)
    
    return unique_terms[:5]

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