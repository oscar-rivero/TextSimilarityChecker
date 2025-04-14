"""
Semantic comparison module for plagiarism detection.
Implementation based on the algorithms from https://github.com/oscar-rivero/plagiarism-detection-prototype-1
"""

import logging
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
import string
import re
import math
from collections import Counter
from bs4 import BeautifulSoup

# Create logger
logger = logging.getLogger(__name__)

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))

def get_wordnet_pos(tag):
    """
    Map POS tag to WordNet POS tag for lemmatization
    """
    if tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return None  # no specific tag

def lemmatize_text(text):
    """
    Tokenize, POS tag, and lemmatize text
    """
    try:
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # POS tag
        tagged_tokens = nltk.pos_tag(tokens)
        
        # Lemmatize according to POS
        lemmas = []
        for word, tag in tagged_tokens:
            try:
                pos = get_wordnet_pos(tag)
                if pos:
                    lemma = lemmatizer.lemmatize(word, pos=pos)
                else:
                    lemma = lemmatizer.lemmatize(word)
                
                # Add non-stopwords to the lemmas list
                if lemma.lower() not in stopwords and len(lemma) > 1:
                    lemmas.append(lemma.lower())
            except Exception as e:
                logger.warning(f"Error lemmatizing word '{word}': {str(e)}")
                # Fall back to the original word if lemmatization fails
                if word.lower() not in stopwords and len(word) > 1:
                    lemmas.append(word.lower())
                
        return lemmas
    except Exception as e:
        logger.error(f"Error in lemmatize_text: {str(e)}")
        # Fall back to simple tokenization
        return [w.lower() for w in text.lower().split() if w.lower() not in stopwords and len(w) > 1]

def find_synonyms(word):
    """
    Find all synonyms for a word using WordNet
    """
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            try:
                for lemma in syn.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if synonym != word:
                        synonyms.add(synonym)
            except Exception as e:
                logger.warning(f"Error getting lemmas for '{word}': {str(e)}")
                continue
    except Exception as e:
        logger.warning(f"Error getting synsets for '{word}': {str(e)}")
    return list(synonyms)

def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts
    Returns a tuple of (similarity_score, semantic_matches)
    Based on techniques from https://github.com/oscar-rivero/plagiarism-detection-prototype-1
    """
    try:
        # Get lemmas for both texts
        lemmas1 = lemmatize_text(text1)
        lemmas2 = lemmatize_text(text2)
        
        # Convert to sets for faster intersection calculations
        lemmas1_set = set(lemmas1)
        lemmas2_set = set(lemmas2)
        
        # Find direct matches (words that appear in both texts)
        direct_matches = lemmas1_set.intersection(lemmas2_set)
        
        # Find semantic matches (synonyms)
        semantic_matches = []
        
        # Words that are unique to each text
        unique_to_text1 = lemmas1_set - lemmas2_set
        unique_to_text2 = lemmas2_set - lemmas1_set
        
        # For each unique word in text1, find if it has synonyms in text2's unique words
        for word1 in unique_to_text1:
            if len(word1) <= 2:  # Skip very short words
                continue
                
            # Get all WordNet synsets for this word
            synsets1 = wordnet.synsets(word1)
            synonyms1 = set()
            hypernyms1 = set()
            hyponyms1 = set()
            
            # Collect all synonyms, hypernyms, and hyponyms
            for syn in synsets1:
                # Add synonyms (lemma names) - with error handling
                try:
                    for lemma in syn.lemmas():
                        synonyms1.add(lemma.name().lower())
                except Exception as e:
                    logger.warning(f"Error getting lemmas for synonyms: {str(e)}")
                    continue
                
                # Add hypernyms (more general terms) - with error handling
                try:
                    for hypernym in syn.hypernyms():
                        for lemma in hypernym.lemmas():
                            hypernyms1.add(lemma.name().lower())
                except Exception as e:
                    logger.warning(f"Error getting hypernyms: {str(e)}")
                
                # Add hyponyms (more specific terms) - with error handling
                try:
                    for hyponym in syn.hyponyms():
                        for lemma in hyponym.lemmas():
                            hyponyms1.add(lemma.name().lower())
                except Exception as e:
                    logger.warning(f"Error getting hyponyms: {str(e)}")
            
            # Remove underscores from multi-word terms
            synonyms1 = {s.replace('_', ' ') for s in synonyms1}
            hypernyms1 = {h.replace('_', ' ') for h in hypernyms1}
            hyponyms1 = {h.replace('_', ' ') for h in hyponyms1}
            
            # Check for matches in the second text
            for word2 in unique_to_text2:
                if len(word2) <= 2:  # Skip very short words
                    continue
                
                # Direct synonym match (highest importance)
                if word2 in synonyms1:
                    semantic_matches.append((word1, word2))
                    continue
                
                # Check for semantic relationship through WordNet
                synsets2 = wordnet.synsets(word2)
                
                for syn2 in synsets2:
                    # Check if they share a common hypernym - with error handling
                    try:
                        for hypernym in syn2.hypernyms():
                            hypernym_lemmas = {lemma.name().lower().replace('_', ' ') for lemma in hypernym.lemmas()}
                            
                            # If they share a common hypernym, they're related
                            if any(h in hypernym_lemmas for h in hypernyms1):
                                semantic_matches.append((word1, word2))
                                break
                    except Exception as e:
                        logger.warning(f"Error checking hypernyms for semantic relation: {str(e)}")
                    
                    # Check if word2's synsets match word1's hypernyms or hyponyms
                    for lemma in syn2.lemmas():
                        lemma_name = lemma.name().lower().replace('_', ' ')
                        if lemma_name in hypernyms1 or lemma_name in hyponyms1:
                            semantic_matches.append((word1, word2))
                            break
        
        # Calculate similarity score using a weighted approach
        direct_match_weight = 1.0
        semantic_match_weight = 0.7
        
        weighted_match_count = (len(direct_matches) * direct_match_weight) + (len(semantic_matches) * semantic_match_weight)
        
        total_unique_words = len(lemmas1_set.union(lemmas2_set))
        if total_unique_words == 0:
            return 0, []
            
        similarity_score = (weighted_match_count / total_unique_words) * 100
        
        # Remove duplicates from semantic matches
        unique_semantic_matches = []
        seen_pairs = set()
        
        for pair in semantic_matches:
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_semantic_matches.append(pair)
                
        return similarity_score, unique_semantic_matches
        
    except Exception as e:
        logger.error(f"Error in semantic comparison: {str(e)}")
        return 0, []

def find_semantic_matching_phrases(text1, text2, min_length=4):
    """
    Find semantically similar phrases between two texts
    Returns a list of matching phrase pairs
    Based on techniques from https://github.com/oscar-rivero/plagiarism-detection-prototype-1
    """
    try:
        # Tokenize both texts into sentences
        sentences1 = nltk.sent_tokenize(text1)
        sentences2 = nltk.sent_tokenize(text2)
        
        matching_phrases = []
        
        # Compare each sentence in text1 with each sentence in text2
        for sent1 in sentences1:
            # Skip very short sentences or sentences without enough substance
            word_count1 = len(sent1.split())
            if word_count1 < min_length or len(sent1) < 20:
                continue
                
            # Get POS tags for the sentence
            tokens1 = word_tokenize(sent1.lower())
            pos_tags1 = nltk.pos_tag(tokens1)
            
            # Check if the sentence has enough substantial words (4+ chars)
            substantial_words = sum(1 for word in tokens1 if len(word) >= 4)
            if substantial_words < 2:
                continue
            
            # Create dictionaries of different POS types
            nouns1 = [word.lower() for word, pos in pos_tags1 if pos.startswith('NN')]
            verbs1 = [word.lower() for word, pos in pos_tags1 if pos.startswith('VB')]
            adjs1 = [word.lower() for word, pos in pos_tags1 if pos.startswith('JJ')]
            
            # Check if the sentence has enough meaningful parts of speech
            if len(nouns1) == 0 or len(verbs1) == 0:
                continue
            
            # Lemmatize the words
            lemmas1 = lemmatize_text(sent1)
            
            for sent2 in sentences2:
                word_count2 = len(sent2.split())
                if word_count2 < min_length or len(sent2) < 20:
                    continue
                
                # Get POS tags for the second sentence
                tokens2 = word_tokenize(sent2.lower())
                pos_tags2 = nltk.pos_tag(tokens2)
                
                # Check if the sentence has enough substantial words
                substantial_words2 = sum(1 for word in tokens2 if len(word) >= 4)
                if substantial_words2 < 2:
                    continue
                
                # Create dictionaries for the second sentence
                nouns2 = [word.lower() for word, pos in pos_tags2 if pos.startswith('NN')]
                verbs2 = [word.lower() for word, pos in pos_tags2 if pos.startswith('VB')]
                adjs2 = [word.lower() for word, pos in pos_tags2 if pos.startswith('JJ')]
                
                # Check if the sentence has enough meaningful parts of speech
                if len(nouns2) == 0 or len(verbs2) == 0:
                    continue
                
                # Lemmatize the second sentence
                lemmas2 = lemmatize_text(sent2)
                
                # Direct word matches (important for establishing similarity)
                direct_matches = set(lemmas1).intersection(set(lemmas2))
                
                # Skip if there aren't enough direct matches to be meaningful
                if len(direct_matches) < 2:
                    continue
                
                # Find semantic matches between specific POS types (nouns, verbs, adjectives)
                semantic_matches = []
                
                # Check nouns against nouns (focus on substantial nouns)
                for noun1 in nouns1:
                    if len(noun1) <= 3:  # Increased minimum length for nouns
                        continue
                        
                    try:
                        synsets1 = wordnet.synsets(noun1, pos=wordnet.NOUN)
                        
                        if not synsets1:
                            continue
                            
                        # Get all synonyms
                        synonyms1 = set()
                        for syn in synsets1:
                            try:
                                for lemma in syn.lemmas():
                                    synonyms1.add(lemma.name().lower().replace('_', ' '))
                            except Exception as e:
                                logger.warning(f"Error processing noun lemmas: {str(e)}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error getting noun synsets: {str(e)}")
                        continue
                    
                    # Check if any noun in the second text is a synonym
                    for noun2 in nouns2:
                        if len(noun2) <= 3:  # Increased minimum length for nouns
                            continue
                            
                        if noun2 in synonyms1:
                            semantic_matches.append((noun1, noun2))
                
                # Check verbs against verbs (focus on substantial verbs)
                for verb1 in verbs1:
                    if len(verb1) <= 3:  # Increased minimum length for verbs
                        continue
                        
                    try:
                        synsets1 = wordnet.synsets(verb1, pos=wordnet.VERB)
                        
                        if not synsets1:
                            continue
                            
                        # Get all synonyms
                        synonyms1 = set()
                        for syn in synsets1:
                            try:
                                for lemma in syn.lemmas():
                                    synonyms1.add(lemma.name().lower().replace('_', ' '))
                            except Exception as e:
                                logger.warning(f"Error processing verb lemmas: {str(e)}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error getting verb synsets: {str(e)}")
                        continue
                    
                    # Check if any verb in the second text is a synonym
                    for verb2 in verbs2:
                        if len(verb2) <= 3:  # Increased minimum length for verbs
                            continue
                            
                        if verb2 in synonyms1:
                            semantic_matches.append((verb1, verb2))
                
                # Calculate similarity score
                # Consider more weight to matching nouns (subjects/objects) and verbs (actions)
                noun_match_ratio = len(set(nouns1).intersection(set(nouns2))) / max(len(nouns1), len(nouns2), 1)
                verb_match_ratio = len(set(verbs1).intersection(set(verbs2))) / max(len(verbs1), len(verbs2), 1)
                
                # Calculate overall similarity
                direct_match_ratio = len(direct_matches) / max(len(lemmas1), len(lemmas2), 1)
                semantic_match_ratio = len(semantic_matches) / max(len(lemmas1), len(lemmas2), 1)
                
                # Weighted similarity score - giving more weight to the structure-defining elements
                similarity = (direct_match_ratio * 50) + (semantic_match_ratio * 30) + (noun_match_ratio * 10) + (verb_match_ratio * 10)
                
                # If similarity is high enough, add it to matching phrases
                if similarity > 30:  # Increased threshold for more meaningful matches
                    # Format the matches for display
                    matching_phrases.append({
                        "original": sent1,
                        "source": sent2,  # Changed to 'source' to match the format in find_matching_phrases
                        "similarity": similarity,
                        "semantic_matches": semantic_matches,
                        "word_count": word_count1  # Add word count for filtering/sorting
                    })
        
        # Sort by similarity score (highest first) and word count (longer sentences first)
        matching_phrases.sort(key=lambda x: (x["similarity"], x["word_count"]), reverse=True)
        
        # Take top matches to avoid overwhelming with too many results
        return matching_phrases[:5]  # Reduced to top 5 for higher quality matches
        
    except Exception as e:
        logger.error(f"Error finding semantic matching phrases: {str(e)}")
        return []

def compare_text_with_paragraphs(input_text, source_html_text, min_paragraph_length=100):
    """
    Compare input text with each paragraph of the source HTML.
    
    This function is based on the buscar_para() function from 
    https://github.com/oscar-rivero/plagiarism-detection-prototype-1/blob/master/modulo_comparador.py
    
    Parameters:
    - input_text: The original text to check for plagiarism
    - source_html_text: The HTML content from a relevant source
    - min_paragraph_length: Minimum character length for paragraphs to consider
    
    Returns:
    - Dictionary with similarity score and matched paragraphs
    """
    try:
        if not source_html_text or not input_text:
            return {
                'overall_similarity': 0,
                'paragraph_matches': []
            }
        
        # Basic sanitization to extract text from HTML
        # Look for paragraphs (text between tags, line breaks, etc.)
        import re
        
        try:
            # Try to handle as plain text with basic cleanup
            # Remove HTML tags if they exist
            clean_source = re.sub(r'<[^>]*>', ' ', source_html_text)
            # Remove extra whitespace
            clean_source = re.sub(r'\s+', ' ', clean_source)
        except Exception as e:
            logger.warning(f"HTML cleaning failed: {e}, using raw text")
            # Use the original text as fallback
            clean_source = source_html_text
        
        # Split into paragraphs - look for common paragraph delimiters
        # 1. Double line breaks
        # 2. HTML paragraph markers (even if tags were removed)
        # 3. Sentence breaks followed by space characters
        paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', clean_source)
        
        # Filter out too short paragraphs and normalize whitespace
        paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
        paragraphs = [re.sub(r'\s+', ' ', p) for p in paragraphs if len(p) >= min_paragraph_length]
        
        # Get the lemmatized input text
        input_lemmas = lemmatize_text(input_text)
        
        paragraph_matches = []
        max_similarity = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Skip empty or too short paragraphs
            if not paragraph or len(paragraph) < min_paragraph_length:
                continue
                
            # Calculate both direct and semantic similarity
            para_lemmas = lemmatize_text(paragraph)
            
            # Direct text similarity using SequenceMatcher
            direct_similarity = SequenceMatcher(None, input_text.lower(), paragraph.lower()).ratio()
            
            # Semantic similarity using our algorithm
            semantic_similarity, matching_pairs = calculate_semantic_similarity(input_text, paragraph)
            
            # Combine similarities - weigh direct matches higher
            combined_similarity = (direct_similarity * 0.7) + (semantic_similarity * 0.3)
            
            if combined_similarity > 0.15:  # Only track significant matches
                paragraph_matches.append({
                    'paragraph_index': i,
                    'paragraph_text': paragraph,
                    'direct_similarity': direct_similarity * 100,  # Convert to percentage
                    'semantic_similarity': semantic_similarity,  # Already a percentage
                    'combined_similarity': combined_similarity * 100,  # Convert to percentage
                    'semantic_matches': matching_pairs
                })
                
                # Keep track of maximum similarity across paragraphs
                if combined_similarity > max_similarity:
                    max_similarity = combined_similarity
        
        # Sort matches by combined similarity
        paragraph_matches.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        # Calculate overall document similarity - weighted by paragraph length
        total_length = sum(len(p['paragraph_text']) for p in paragraph_matches) if paragraph_matches else 1
        weighted_similarity = sum(p['combined_similarity'] * len(p['paragraph_text']) / total_length 
                                for p in paragraph_matches) if paragraph_matches else 0
        
        # Boost similarity if multiple paragraphs match
        paragraph_count_boost = min(1.0, len(paragraph_matches) / 5) * 0.3
        overall_similarity = (max_similarity * 0.7) + (weighted_similarity * 0.3) + paragraph_count_boost
        
        # Cap at 100%
        overall_similarity = min(1.0, overall_similarity) * 100
        
        return {
            'overall_similarity': overall_similarity,
            'paragraph_matches': paragraph_matches[:5]  # Limit to top 5 matches
        }
        
    except Exception as e:
        logger.error(f"Error in paragraph comparison: {str(e)}")
        return {
            'overall_similarity': 0,
            'paragraph_matches': []
        }

def calculate_cosine_similarity(text1, text2):
    """
    Calculate cosine similarity between two texts.
    
    Parameters:
    - text1: First text for comparison
    - text2: Second text for comparison
    
    Returns:
    - A float between 0 and 1 representing the cosine similarity
    """
    try:
        # Preprocess texts
        lemmas1 = lemmatize_text(text1)
        lemmas2 = lemmatize_text(text2)
        
        # Create term frequency dictionaries
        vector1 = Counter(lemmas1)
        vector2 = Counter(lemmas2)
        
        # Calculate dot product
        dot_product = sum(vector1[term] * vector2[term] for term in set(vector1).intersection(set(vector2)))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
        magnitude2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        # Return cosine similarity (dot product / product of magnitudes)
        return dot_product / (magnitude1 * magnitude2)
    
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0

def extract_paragraphs_from_source(source_text, min_length=50):
    """
    Extract paragraphs from source text.
    
    Parameters:
    - source_text: Text to extract paragraphs from
    - min_length: Minimum character length for a valid paragraph
    
    Returns:
    - List of extracted paragraphs
    """
    try:
        # Try to extract paragraphs from HTML first
        try:
            # Use BeautifulSoup for more reliable HTML parsing
            soup = BeautifulSoup(source_text, 'html.parser')
            paragraphs = []
            
            # Extract all paragraphs
            for p_tag in soup.find_all('p'):
                p_text = p_tag.get_text(strip=True)
                if p_text and len(p_text) >= min_length:
                    paragraphs.append(p_text)
            
            # If we found paragraphs, return them
            if paragraphs:
                return paragraphs
        except Exception as e:
            logger.warning(f"Error extracting paragraphs from HTML: {str(e)}")
        
        # Fallback to text-based extraction
        # Split on double newlines (common paragraph separator)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', source_text)]
        
        # If we still don't have paragraphs, try single newlines
        if not paragraphs or all(len(p) < min_length for p in paragraphs):
            paragraphs = [p.strip() for p in re.split(r'\n', source_text)]
        
        # If we still don't have paragraphs, split on periods followed by spaces
        if not paragraphs or all(len(p) < min_length for p in paragraphs):
            paragraphs = [p.strip() + '.' for p in re.split(r'\.(?=\s)', source_text) if p.strip()]
        
        # Filter out too-short paragraphs
        return [p for p in paragraphs if len(p) >= min_length]
    
    except Exception as e:
        logger.error(f"Error extracting paragraphs: {str(e)}")
        # Return the whole text as one paragraph as a fallback
        return [source_text] if source_text and len(source_text) >= min_length else []

def find_best_matching_paragraph(input_text, source_text):
    """
    Find the paragraph from the source text that best matches the input text,
    evaluated with cosine similarity.
    
    Parameters:
    - input_text: The text to check against source paragraphs
    - source_text: The source text to extract paragraphs from
    
    Returns:
    - Dictionary with the best matching paragraph and its similarity score
    """
    try:
        # Extract paragraphs from source
        paragraphs = extract_paragraphs_from_source(source_text)
        
        if not paragraphs:
            return {
                'paragraph': '',
                'similarity': 0
            }
        
        # Calculate cosine similarity for each paragraph
        similarities = []
        for paragraph in paragraphs:
            similarity = calculate_cosine_similarity(input_text, paragraph)
            similarities.append({
                'paragraph': paragraph,
                'similarity': similarity
            })
        
        # Return the paragraph with the highest similarity
        if similarities:
            return max(similarities, key=lambda x: x['similarity'])
        else:
            return {
                'paragraph': '',
                'similarity': 0
            }
    
    except Exception as e:
        logger.error(f"Error finding best matching paragraph: {str(e)}")
        return {
            'paragraph': '',
            'similarity': 0
        }

def check_semantic_plagiarism(original_text, source_texts):
    """
    Check for semantic plagiarism between the original text and a list of source texts
    Returns a list of results with similarity scores and matching phrases
    
    Now includes paragraph-level comparison for more precise detection
    based on techniques from https://github.com/oscar-rivero/plagiarism-detection-prototype-1
    """
    results = []
    
    for source in source_texts:
        try:
            # Skip sources without content
            source_content = source.get("content", "")
            if not source_content or len(source_content.strip()) < 50:
                logger.warning(f"Skipping source with insufficient content: {source.get('url', 'unknown')}")
                continue
                
            # Get needed fields with defaults
            title = source.get("title", "Untitled Source")
            url = source.get("url", "#")
            snippet = source.get("snippet", "")
            
            # We'll determine if this is a Wikipedia article based on URL
            is_wikipedia = "wikipedia.org" in url.lower()
            # Get categories for this source
            categories = source.get("categories", [])
            # Convert to list if it's a string
            if isinstance(categories, str):
                categories = [categories]
            # Get relevance score
            relevance_score = source.get("relevance_score", 0)
            
            try:
                # Calculate overall semantic similarity score
                similarity, semantic_matches = calculate_semantic_similarity(original_text, source_content)
                
                # Find matching phrases
                matching_phrases = find_semantic_matching_phrases(original_text, source_content)
                
                # New: Perform the paragraph-level comparison
                paragraph_comparison = compare_text_with_paragraphs(original_text, source_content)
                paragraph_similarity = paragraph_comparison['overall_similarity']
                paragraph_matches = paragraph_comparison['paragraph_matches']
                
                # Use the maximum similarity score between methods
                # Weight paragraph comparison higher for historical/academic content
                final_similarity = similarity
                if "history" in categories or "literature" in categories or "education" in categories:
                    # For historical or academic content, paragraph comparison is more relevant
                    final_similarity = max(similarity, paragraph_similarity * 1.2)
                else:
                    # For other content, use the higher of the two methods
                    final_similarity = max(similarity, paragraph_similarity)
                
                # Boost Wikipedia sources for exact topic matches
                if is_wikipedia and relevance_score > 1.0:
                    # Boost Wikipedia articles that are very relevant
                    final_similarity *= 1.3
                
                if final_similarity > 10 or matching_phrases or paragraph_matches:  # Lower threshold for comparison
                    result = {
                        "source": {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "is_wikipedia": is_wikipedia,
                            "categories": categories,
                            "relevance_score": relevance_score
                        },
                        "similarity": final_similarity,
                        "matches": matching_phrases,
                        "semantic_matches": semantic_matches,
                        "paragraph_matches": paragraph_matches
                    }
                    results.append(result)
            except ValueError as e:
                logger.error(f"Value error in semantic comparison for {url}: {str(e)}")
                continue
            except TypeError as e:
                logger.error(f"Type error in semantic comparison for {url}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing source {url}: {str(e)}")
                continue
                
        except Exception as e:
            logger.error(f"Error accessing source data: {str(e)}")
    
    # New improved sorting - give more weight to Wikipedia articles and high relevance sources
    # This is critical for ensuring the most relevant results appear first
    def source_ranking_score(result):
        # Start with the similarity score
        score = result["similarity"]
        
        # Boost Wikipedia sources
        if result["source"].get("is_wikipedia", False):
            score *= 1.5
        
        # Boost sources with high relevance scores
        relevance = result["source"].get("relevance_score", 0)
        if relevance > 1.0:
            score *= (1 + (relevance - 1) * 0.3)
            
        # Boost sources with paragraph matches
        if result.get("paragraph_matches") and len(result["paragraph_matches"]) > 0:
            score *= (1 + min(len(result["paragraph_matches"]), 5) * 0.1)
            
        return score
    
    # Sort results by our custom ranking score
    results.sort(key=source_ranking_score, reverse=True)
    
    return results