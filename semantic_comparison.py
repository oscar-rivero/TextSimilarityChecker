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
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # POS tag
    tagged_tokens = nltk.pos_tag(tokens)
    
    # Lemmatize according to POS
    lemmas = []
    for word, tag in tagged_tokens:
        pos = get_wordnet_pos(tag)
        if pos:
            lemma = lemmatizer.lemmatize(word, pos=pos)
        else:
            lemma = lemmatizer.lemmatize(word)
        
        # Add non-stopwords to the lemmas list
        if lemma.lower() not in stopwords and len(lemma) > 1:
            lemmas.append(lemma.lower())
            
    return lemmas

def find_synonyms(word):
    """
    Find all synonyms for a word using WordNet
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().lower().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
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
                # Add synonyms (lemma names)
                for lemma in syn.lemmas():
                    synonyms1.add(lemma.name().lower())
                
                # Add hypernyms (more general terms)
                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        hypernyms1.add(lemma.name().lower())
                
                # Add hyponyms (more specific terms)
                for hyponym in syn.hyponyms():
                    for lemma in hyponym.lemmas():
                        hyponyms1.add(lemma.name().lower())
            
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
                    # Check if they share a common hypernym
                    for hypernym in syn2.hypernyms():
                        hypernym_lemmas = {lemma.name().lower().replace('_', ' ') for lemma in hypernym.lemmas()}
                        
                        # If they share a common hypernym, they're related
                        if any(h in hypernym_lemmas for h in hypernyms1):
                            semantic_matches.append((word1, word2))
                            break
                    
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
                        
                    synsets1 = wordnet.synsets(noun1, pos=wordnet.NOUN)
                    
                    if not synsets1:
                        continue
                        
                    # Get all synonyms
                    synonyms1 = set()
                    for syn in synsets1:
                        for lemma in syn.lemmas():
                            synonyms1.add(lemma.name().lower().replace('_', ' '))
                    
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
                        
                    synsets1 = wordnet.synsets(verb1, pos=wordnet.VERB)
                    
                    if not synsets1:
                        continue
                        
                    # Get all synonyms
                    synonyms1 = set()
                    for syn in synsets1:
                        for lemma in syn.lemmas():
                            synonyms1.add(lemma.name().lower().replace('_', ' '))
                    
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

def check_semantic_plagiarism(original_text, source_texts):
    """
    Check for semantic plagiarism between the original text and a list of source texts
    Returns a list of results with similarity scores and matching phrases
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
            
            try:
                # Calculate similarity score
                similarity, semantic_matches = calculate_semantic_similarity(original_text, source_content)
                
                # Find matching phrases
                matching_phrases = find_semantic_matching_phrases(original_text, source_content)
                
                if similarity > 10 or matching_phrases:  # Lower threshold for semantic comparison
                    result = {
                        "source": {
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        },
                        "similarity": similarity,
                        "matches": matching_phrases,
                        "semantic_matches": semantic_matches
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
    
    # Sort results by similarity (descending)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results