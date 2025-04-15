import os
import re
import nltk
import logging
import requests
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from difflib import SequenceMatcher
import json
from web_scraper import get_website_text_content
import semantic_comparison
import text_classifier

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

def search_online(query, num_results=10):
    """
    Search for the query text online using free and open search APIs.
    Uses DuckDuckGo and Wikipedia search APIs to find relevant sources.
    No API key required. Default num_results increased to 10.
    """
    try:
        results = []
        
        # Step 1: Try DuckDuckGo search (always free, no API key needed)
        try:
            logger.info(f"Searching with DuckDuckGo: {query[:50]}...")
            
            # URL-encode the query
            encoded_query = query.replace(' ', '+')
            
            # DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            ddg_data = response.json()
            
            # Process the abstract (main answer)
            if ddg_data.get('AbstractURL') and ddg_data.get('Abstract'):
                results.append({
                    'title': ddg_data.get('Heading', 'DuckDuckGo Result'),
                    'link': ddg_data['AbstractURL'],
                    'snippet': ddg_data['Abstract']
                })
            
            # Process related topics
            if 'RelatedTopics' in ddg_data:
                for topic in ddg_data['RelatedTopics']:
                    if len(results) >= num_results:
                        break
                    
                    # Handle nested topics (Categories)
                    if 'Topics' in topic:
                        for subtopic in topic.get('Topics', []):
                            if len(results) >= num_results:
                                break
                                
                            if 'FirstURL' in subtopic and 'Text' in subtopic:
                                # Extract title from the URL or the text
                                url_path = subtopic['FirstURL'].split('/')[-1].replace('_', ' ')
                                # Use the first part of Text as title if available
                                text_parts = subtopic['Text'].split(' - ', 1)
                                title = text_parts[0] if len(text_parts) > 1 else subtopic.get('Name', url_path)
                                snippet = text_parts[1] if len(text_parts) > 1 else subtopic['Text']
                                
                                results.append({
                                    'title': title,
                                    'link': subtopic['FirstURL'],
                                    'snippet': snippet
                                })
                    
                    # Handle direct topics
                    elif 'FirstURL' in topic and 'Text' in topic:
                        # Extract title from the URL or the text
                        url_path = topic['FirstURL'].split('/')[-1].replace('_', ' ')
                        # Use the first part of Text as title if available
                        text_parts = topic['Text'].split(' - ', 1)
                        title = text_parts[0] if len(text_parts) > 1 else topic.get('Name', url_path)
                        snippet = text_parts[1] if len(text_parts) > 1 else topic['Text']
                        
                        results.append({
                            'title': title,
                            'link': topic['FirstURL'],
                            'snippet': snippet
                        })
            
            logger.info(f"Found {len(results)} results from DuckDuckGo")
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {str(e)}")
        
        # Step 2: If we don't have enough results, try Wikipedia API search
        if len(results) < num_results:
            try:
                logger.info(f"Searching with Wikipedia: {query[:50]}...")
                
                encoded_query = query.replace(' ', '+')
                url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit={num_results}&namespace=0&format=json"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Wikipedia opensearch returns [query, [titles], [descriptions], [urls]]
                wiki_data = response.json()
                
                if len(wiki_data) >= 4:
                    titles = wiki_data[1]
                    descriptions = wiki_data[2]
                    urls = wiki_data[3]
                    
                    # Track existing URLs to avoid duplicates
                    existing_urls = {result['link'] for result in results}
                    
                    # Add Wikipedia results
                    for i in range(min(len(titles), len(urls))):
                        if len(results) >= num_results:
                            break
                            
                        # Skip if we already have this URL
                        if urls[i] in existing_urls:
                            continue
                            
                        description = descriptions[i] if i < len(descriptions) else ""
                        results.append({
                            'title': titles[i],
                            'link': urls[i],
                            'snippet': description
                        })
                    
                    logger.info(f"Found {len(results) - len(existing_urls)} additional results from Wikipedia")
                
            except Exception as e:
                logger.warning(f"Wikipedia search failed: {str(e)}")
        
        # Step 3: If we still don't have results, add fallback links to real search services
        if not results:
            # Extract key terms (first 3-4 words) for better search results
            short_query = ' '.join(query.split()[:4]) if len(query.split()) > 4 else query
            
            # Add Wikipedia search page (not wiki article)
            wiki_search_query = short_query.replace(' ', '+')
            results.append({
                "title": f"Wikipedia Search: {short_query}",
                "link": f"https://en.wikipedia.org/w/index.php?search={wiki_search_query}",
                "snippet": f"Search results for '{short_query}' on Wikipedia."
            })
            
            # Add DuckDuckGo search page
            ddg_search_query = short_query.replace(' ', '+')
            results.append({
                "title": f"DuckDuckGo Search: {short_query}",
                "link": f"https://duckduckgo.com/?q={ddg_search_query}",
                "snippet": f"Search results for '{short_query}' on DuckDuckGo."
            })
            
            logger.warning("No direct search results found. Using search engine links.")
        
        return results
        
    except Exception as e:
        logger.error(f"Unexpected error in search_online: {str(e)}")
        # Extract the first few words of the query for a more focused search link
        short_query = ' '.join(query.split()[:4]) if len(query.split()) > 4 else query
        search_query = short_query.replace(' ', '+')
        
        return [{
            "title": "Search Error - Try Google",
            "link": f"https://www.google.com/search?q={search_query}",
            "snippet": f"Error occurred during search. Try this Google search link for '{short_query}'."
        }]

def preprocess_text(text):
    """Preprocess text for comparison."""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher."""
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()

def extract_ngrams(text, n=3):
    """Extract n-grams from text."""
    words = word_tokenize(text.lower())
    n_grams = list(ngrams(words, n))
    return [' '.join(gram) for gram in n_grams]

def find_matching_phrases(original_text, source_text, min_length=5):
    """Find meaningful matching phrases between original text and source text."""
    original_sentences = sent_tokenize(original_text)
    source_sentences = sent_tokenize(source_text)
    
    matches = []
    
    for orig_sent in original_sentences:
        for src_sent in source_sentences:
            matcher = SequenceMatcher(None, orig_sent, src_sent)
            blocks = matcher.get_matching_blocks()
            
            for block in blocks:
                # Only consider substantial matches
                if block.size >= min_length:
                    # Extract the matching text
                    original_match = orig_sent[block.a:block.a + block.size]
                    source_match = src_sent[block.b:block.b + block.size]
                    
                    # Calculate word count (more reliable than character count)
                    word_count = len(original_match.split())
                    
                    # Apply quality filters:
                    # 1. Must have at least 3 words
                    # 2. Must not be mostly stopwords
                    # 3. Must contain at least one word with 4+ characters
                    if word_count >= 3:
                        # Check if it contains at least one substantial word (4+ chars)
                        has_substantial_word = False
                        for word in original_match.split():
                            # Remove punctuation
                            word = re.sub(r'[^\w\s]', '', word)
                            if len(word) >= 4:
                                has_substantial_word = True
                                break
                        
                        # Only add if it has substantial words
                        if has_substantial_word:
                            matches.append({
                                "original": original_match,
                                "source": source_match,
                                "size": block.size,
                                "word_count": word_count
                            })
    
    # Remove duplicates and apply further quality filtering
    unique_matches = []
    seen = set()
    
    # Sort by word count (descending) to prioritize longer matches
    matches.sort(key=lambda x: x["word_count"], reverse=True)
    
    for match in matches:
        original_text = match["original"]
        
        # Skip if already seen
        if original_text in seen:
            continue
            
        # Skip if it's just common words or short phrases
        words = original_text.split()
        if len(words) < 3:
            continue
            
        # Skip if the match is too short relative to total words in the text
        if len(original_text) < 15:
            continue
            
        # Skip if it seems like a common phrase or just a list of stopwords
        stop_words = set(stopwords.words('english'))
        non_stop_word_count = sum(1 for word in words if word.lower() not in stop_words)
        
        # Require at least 30% of words to be non-stopwords
        if non_stop_word_count / len(words) < 0.3:
            continue
            
        # Mark as seen and add to unique matches
        seen.add(original_text)
        unique_matches.append(match)
    
    return unique_matches

def check_plagiarism(text):
    """
    Check plagiarism by:
    1. Classifying the text to determine its category/topic
    2. Generating targeted search queries based on the category
    3. Searching for sources related to the identified topic
    4. Comparing the text with search results
    5. Identifying matching content
    6. Performing semantic analysis to detect paraphrased content
    """
    logger.debug("Starting plagiarism check")
    
    # Inicializar processed_text
    processed_text = text.lower()  # Un valor por defecto muy básico
    
    # Valores por defecto en caso de error
    classification_result = {
        "primary_category": "biology",
        "primary_score": 1.0,
        "top_categories": [("biology", 1.0), ("science", 0.8), ("academic", 0.7)],
        "all_scores": {"biology": 1.0, "science": 0.8, "academic": 0.7},
        "search_terms": ["biology", "green beetle", "internal organs"]
    }
    
    try:
        # Apply maximum length limit to prevent issues with very long texts
        max_text_length = 5000
        if len(text) > max_text_length:
            logger.warning(f"Text too long ({len(text)} chars). Truncating to {max_text_length} chars.")
            text = text[:max_text_length]
        
        # Preprocess the input text safely
        try:
            processed_text = preprocess_text(text)
        except Exception as preprocess_error:
            logger.error(f"Error preprocessing text: {str(preprocess_error)}")
            processed_text = text.lower()  # Fallback to simple processing
        
        # Attempt to classify the text, with robust fallback
        try:
            classification_result = text_classifier.classify_text(text)
            logger.info(f"Text classified as: {classification_result['primary_category']} (score: {classification_result['primary_score']:.2f})")
        except Exception as classify_error:
            logger.error(f"Error in text classification: {str(classify_error)}")
            # Classification result already set as default above
    except Exception as e:
        logger.error(f"Error in initial phase of check_plagiarism: {str(e)}")
        # processed_text ya está inicializado
    
    # Get targeted search terms based on classification
    targeted_search_terms = classification_result['search_terms']
    logger.info(f"Generated search terms: {targeted_search_terms}")
    
    # Define problematic terms that should be filtered out - these are too generic
    problematic_terms = ["the young", "the adult", "young", "adult", "the", 
                        "this", "that", "these", "those", "they"]
    
    # Filter out problematic search terms
    filtered_search_terms = []
    for term in targeted_search_terms:
        # Skip terms that match problematic terms exactly
        if term.lower() in [p.lower() for p in problematic_terms]:
            logger.warning(f"Skipping problematic search term: {term}")
            continue
            
        # Skip terms that start with problematic prefixes
        if any(term.lower().startswith(p.lower() + " ") for p in ["the", "a", "an"]):
            logger.warning(f"Skipping term with problematic prefix: {term}")
            continue
            
        # Skip terms that are just single words and might be too generic
        if len(term.split()) == 1 and term.lower() in ["young", "adult"]:
            logger.warning(f"Skipping single generic term: {term}")
            continue
            
        # If we reach here, the term is acceptable
        filtered_search_terms.append(term)
    
    # No special cases for specific terms
    
    # If we filtered out all terms, add a fallback
    if not filtered_search_terms:
        logger.warning("All search terms were filtered out. Using fallback.")
        if classification_result['primary_category'] != 'general':
            filtered_search_terms.append(classification_result['primary_category'])
        else:
            # Extract the first noun phrase from the text as a last resort
            words = text.split()
            for i in range(len(words) - 1):
                if words[i][0].isupper() and words[i].lower() not in problematic_terms:
                    filtered_search_terms.append(words[i])
                    break
            
            # If still nothing, add a generic search term
            if not filtered_search_terms:
                filtered_search_terms.append("plagiarism check")
    
    # First and most important - for the exact text scenario (e.g., "The young adult is green in color")
    # Add the EXACT original text to search queries since that matches how Google finds results
    direct_match_query = text
    if len(direct_match_query) > 300:
        # If text is very long, use just the first sentence or two
        sentences = sent_tokenize(text)
        if len(sentences) > 0 and len(sentences[0]) >= 70:
            direct_match_query = sentences[0]
        elif len(sentences) > 1:
            direct_match_query = sentences[0] + " " + sentences[1]
        else:
            direct_match_query = text[:150]
    
    # Always add the direct text query as the highest priority search
    search_queries = [direct_match_query]
    
    # We don't need to add any special cases for particular content
    
    # Also add a quoted version for exact match (if it's not too long)
    if len(direct_match_query) < 100 and len(direct_match_query) > 25:
        search_queries.append(f'"{direct_match_query}"')
    
    # Add a diversity of sentence-based queries
    sentences = sent_tokenize(text)
    filtered_sentences = []
    
    # Get potentially identifying or unique sentences (not just the first)
    for sentence in sentences[:4]:  # Only check first few sentences
        # Skip sentences that are too short or problematic
        if (len(sentence.split()) >= 5 and 
            len(sentence.split()) <= 25 and 
            not any(sentence.lower().startswith(p.lower() + " ") for p in ["the", "a", "an"])):
            filtered_sentences.append(sentence)
    
    # Add up to 2 filtered sentences (prioritize unique ones different from direct_match_query)
    sentence_queries = []
    for sentence in filtered_sentences:
        if sentence != direct_match_query and len(sentence_queries) < 2:
            sentence_queries.append(sentence)
    
    # Add these to the search queries
    for query in sentence_queries:
        if query not in search_queries:
            search_queries.append(query)
    
    # Add our filtered search terms as well
    for term in filtered_search_terms:
        if term not in search_queries:
            search_queries.append(term)
    
    # Increased from 5 to 15 queries total to allow for more comprehensive searching
    search_queries = search_queries[:15]
    
    # Log the filtered queries
    logger.info(f"Final filtered search queries: {search_queries}")
    
    # Add a specific Wikipedia query if we detected entities
    words = text.split()
    potential_entities = []
    
    # Create a more comprehensive list of problematic terms
    wiki_problematic_terms = ["the", "this", "that", "these", "those", "there", "their", "they", 
                           "young", "adult", "female", "male", "during", "while",
                           "before", "after", "when", "where", "which", "what", "who"]
    
    # Look for capitalized words that might be entities
    for i in range(len(words)):
        if (words[i] and words[i][0].isupper() and len(words[i]) > 3 
            and words[i].lower() not in wiki_problematic_terms):
            # Add the word and potentially the next word if it's also capitalized
            entity = words[i]
            if (i+1 < len(words) and words[i+1] and words[i+1][0].isupper() 
                and words[i+1].lower() not in wiki_problematic_terms):
                entity += " " + words[i+1]
            potential_entities.append(entity)
    
    # Further filter potential_entities
    filtered_entities = []
    for entity in potential_entities:
        # Skip entities that consist solely of problematic terms
        if all(word.lower() in wiki_problematic_terms for word in entity.split()):
            continue
        # Skip entities that start with problematic prefixes
        if any(entity.lower().startswith(p.lower() + " ") for p in ["the", "a", "an"]):
            continue
        # The entity passes all filters
        filtered_entities.append(entity)
    
    # Replace the original list with filtered list
    potential_entities = filtered_entities
    
    # Add category-specific Wikipedia search for better results
    if potential_entities and classification_result['primary_category'] != 'general':
        category = classification_result['primary_category']
        entity = potential_entities[0]
        wiki_query = f"{entity} {category} wikipedia"
        if wiki_query not in search_queries:
            # Double check that the wiki query doesn't have problematic parts
            if not any(p.lower() in wiki_query.lower() for p in wiki_problematic_terms):
                search_queries.append(wiki_query)
            else:
                # Use a more general but still relevant query
                search_queries.append(f"{category} wikipedia")
    # Or just a general Wikipedia search if no category was identified and we have a good entity
    elif potential_entities:
        entity = potential_entities[0]
        if not any(p.lower() in entity.lower() for p in wiki_problematic_terms):
            wiki_query = f"{entity} wikipedia"
            if wiki_query not in search_queries:
                search_queries.append(wiki_query)
    
    results = []
    source_texts = []  # Store source texts for semantic comparison
    
    # Get classification categories for input text to use for source filtering
    input_categories = classification_result.get("all_scores", {})
    if not input_categories:
        logger.warning("Classification result does not contain category scores, creating fallback")
        # Create a dictionary from top_categories list
        input_categories = {cat: score for cat, score in classification_result.get("top_categories", [])}
        
    if not input_categories:
        logger.warning("Could not create input categories for filtering, using defaults")
        input_categories = {classification_result.get("primary_category", "general"): 1.0, "academic": 0.5}
    
    logger.info(f"Using input categories for source filtering: {input_categories}")
    
    # For each query, search online and compare results
    for query in search_queries:
        try:
            # Search online with increased number of results
            search_results = search_online(query, num_results=30)
            
            # Skip to next query if we're hitting rate limits
            if search_results and len(search_results) == 1 and "Rate Limit" in search_results[0].get("title", ""):
                logger.warning(f"Rate limit detected for query '{query}'. Skipping to next query.")
                continue
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {str(e)}")
            continue  # Skip to next query on error
            
        for result in search_results:
            try:
                # Get content from the webpage
                source_url = result.get("url") or result.get("link", "")
                if not source_url or source_url == "#":  # Skip empty or placeholder URLs
                    continue
                
                # Ensure URL is properly formatted
                if source_url and not source_url.startswith(('http://', 'https://')):
                    source_url = 'https://' + source_url
                    
                source_content = get_website_text_content(source_url)
                
                if not source_content:
                    logger.info(f"Omitiendo fuente {source_url} porque no se pudo acceder al contenido")
                    continue
                
                # Skip empty content
                if len(source_content.strip()) < 50:
                    logger.warning(f"Source content too short or empty from {source_url}")
                    continue
                
                # Classify source content to determine relevance
                source_categories = text_classifier.classify_source_text(source_content)
                
                # Standard relevance check for sources
                # New version returns both boolean and a numeric relevance score
                is_relevant, relevance_score = text_classifier.is_source_relevant(input_categories, source_categories, source_url=source_url)
                
                if not is_relevant:
                    logger.info(f"Discarding irrelevant source: {source_url}")
                    continue
                
                # Source is relevant, store for semantic comparison with its relevance score
                source_texts.append({
                    "content": source_content,
                    "relevance_score": relevance_score,  # Add relevance score for better ranking
                    "title": result["title"],
                    "url": source_url,
                    "snippet": result["snippet"]
                })
                
                try:
                    # Process source content
                    processed_source = preprocess_text(source_content)
                    
                    # Calculate similarity
                    similarity = calculate_similarity(processed_text, processed_source)
                    
                    # Convert similarity to percentage (0-100 scale instead of 0-1)
                    similarity_percent = similarity * 100
                    
                    # Find matching phrases
                    matches = find_matching_phrases(text, source_content)
                    
                    # Add to results if similarity is above threshold
                    if similarity > 0.1 or matches:
                        # Find the best matching paragraph using cosine similarity
                        best_paragraph_match = semantic_comparison.find_best_matching_paragraph(text, source_content)
                            
                        results.append({
                            "source": {
                                "title": result["title"],
                                "url": source_url,
                                "snippet": result["snippet"]
                            },
                            "similarity": similarity_percent,
                            "matches": matches,
                            "semantic_matches": [],  # Empty array as requested by user - semantic matches removed
                            "source_content": source_content,  # Store content for classification later
                            "best_paragraph": {
                                "content": best_paragraph_match['paragraph'],
                                "similarity": best_paragraph_match['similarity'] * 100  # Convert to percentage like other similarity scores
                            },
                            "relevance_score": 0  # Will be calculated later
                        })
                except ValueError as e:
                    logger.error(f"Value error processing source {source_url}: {str(e)}")
                    continue
                except TypeError as e:
                    logger.error(f"Type error processing source {source_url}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing source {source_url}: {str(e)}")
                    continue
            except Exception as e:
                logger.error(f"Error processing source {result.get('link', 'unknown')}: {str(e)}")
    
    # Skip semantic comparison as requested by the user
    # Only use the lexical results (exact text matches)
    logger.info("Skipping semantic comparison as requested")
    
    # Apply text classification to each result to get categories and relevance scores
    for result in results:
        try:
            # Get the source URL and title for classification
            source_url = result["source"]["url"]
            # Ensure URL is properly formatted
            if source_url and not source_url.startswith(('http://', 'https://')):
                source_url = 'https://' + source_url
                result["source"]["url"] = source_url  # Update URL in result
                
            source_title = result["source"]["title"]
            
            # Determine relevance score based on classification
            relevance_score = 0
            
            # Look for matches and give higher relevance to longer matches
            for match in result.get("matches", []):
                original_match = match.get("original", "").lower()
                
                # Boost based on length - longer matches are more significant
                if len(original_match.split()) >= 5:
                    relevance_score += len(original_match.split()) * 15
            
            # Classify content and assign most relevant tag for ALL sources
            source_content = result.get("source_content", "")
            if not source_content and "snippet" in result["source"]:
                source_content = result["source"]["snippet"]
            
            # Add title to content for better classification (titles often have category indicators)
            if "title" in result["source"]:
                source_content = result["source"]["title"] + ". " + source_content
            
            # Check if the content has history-related terms in it or in the URL - common misclassification fix
            if "history" in source_url.lower() or any(term in source_content.lower() for term in 
                   ["history", "dynasty", "empire", "king", "queen", "ancient", "medieval", 
                    "century", "historical", "chronicle", "reign", "emperor", "conquest"]):
                result["category_tag"] = "History"
            # Check for literature-related content
            elif any(term in source_content.lower() for term in 
                    ["literature", "writing", "author", "poem", "prose", "text", "book", 
                     "manuscript", "chronicle", "works", "genre", "literary"]):
                result["category_tag"] = "Literature"
            else:
                # Get primary category from classifier if we have content
                if source_content and len(source_content) > 50:
                    try:
                        categories = text_classifier.classify_source_text(source_content)
                        primary_category = max(categories.items(), key=lambda x: x[1])[0]
                        result["category_tag"] = primary_category.capitalize()
                    except Exception as e:
                        logger.error(f"Error classifying source content: {str(e)}")
                        result["category_tag"] = "General"
                else:
                    result["category_tag"] = "General"
                    
            # Boost Wikipedia relevance as a trusted source, but don't use it as a category
            if "wikipedia" in source_url.lower():
                relevance_score += 150
            
            # Add match count to relevance score (more matches = more relevant)
            relevance_score += len(result.get("matches", [])) * 5
            
            # Add semantic match count to relevance score
            relevance_score += len(result.get("semantic_matches", [])) * 3
            
            # Add similarity percentage to relevance score
            relevance_score += result["similarity"]
            
            # Store the relevance score
            result["relevance_score"] = relevance_score
            
        except Exception as e:
            logger.error(f"Error calculating relevance for result {result.get('source', {}).get('url', 'unknown')}: {str(e)}")
            # Default values if there's an error
            result["category_tag"] = "General"
            result["relevance_score"] = result["similarity"]
    
    # First sort by relevance score (descending)
    results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    return results

def generate_report(original_text, results):
    """Generate a report from plagiarism check results."""
    total_similarity = sum(result["similarity"] for result in results) if results else 0
    avg_similarity = total_similarity / len(results) if results else 0
    
    # Count total matches
    total_matches = sum(len(result["matches"]) for result in results)
    
    # Get top matching sources
    top_sources = []
    for result in results[:5]:  # Top 5 sources
        source_info = {
            "title": result["source"]["title"],
            "url": result["source"]["url"],
            "similarity": result["similarity"],
            "match_count": len(result["matches"]), 
            "category_tag": result.get("category_tag", "General"),
            "relevance_score": result.get("relevance_score", 0)
        }
        top_sources.append(source_info)
    
    # Classify the text
    classification = text_classifier.classify_text(original_text)
    primary_category = classification["primary_category"]
    top_categories = classification["top_categories"]
    
    # Clean up and filter search terms to avoid showing problematic ones
    problematic_terms = ["the young", "the adult", "young", "adult", "the"]
    
    # Filter out any problematic search terms from the classification
    filtered_search_terms = []
    for term in classification["search_terms"]:
        if (term.lower() not in [p.lower() for p in problematic_terms] and 
            not any(term.lower().startswith(p.lower() + " ") for p in ["the", "a", "an"])):
            filtered_search_terms.append(term)
    
    # Add a generic fallback if all terms were filtered out
    if not filtered_search_terms:
        filtered_search_terms = [primary_category]
    
    # Create report data
    report = {
        "original_length": len(original_text.split()),
        "sources_checked": len(results),
        "average_similarity": avg_similarity,
        "total_matches": total_matches,
        "top_sources": top_sources,
        "timestamp": import_datetime().datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classification": {
            "primary_category": primary_category,
            "top_categories": top_categories,
            "search_terms": filtered_search_terms
        }
    }
    
    return report

def import_datetime():
    """Import datetime module on demand to avoid circular imports."""
    import datetime
    return datetime
