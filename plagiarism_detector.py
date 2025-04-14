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

def search_online(query, num_results=5):
    """
    Search for the query text online using SerpAPI.
    This function should be replaced with an actual implementation using a search API.
    """
    try:
        # Get API key from environment variable
        api_key = os.environ.get("SERPAPI_KEY")
        if not api_key:
            logger.warning("No SERPAPI_KEY found in environment variables. Using limited search functionality.")
            # Return dummy results for testing without API key
            return [
                {"title": "No search API key provided", "link": "#", "snippet": "Add SERPAPI_KEY to environment variables for real search results."}
            ]
        
        # Make request to SerpAPI
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results
        }
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        
        search_results = response.json().get("organic_results", [])
        return [{"title": result.get("title", ""), 
                 "link": result.get("link", ""), 
                 "snippet": result.get("snippet", "")} 
                for result in search_results]
    
    except requests.RequestException as e:
        logger.error(f"Error searching online: {str(e)}")
        return [{"title": "Search API Error", "link": "#", "snippet": f"Error: {str(e)}"}]

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
    """Find matching phrases between original text and source text."""
    original_sentences = sent_tokenize(original_text)
    source_sentences = sent_tokenize(source_text)
    
    matches = []
    
    for orig_sent in original_sentences:
        for src_sent in source_sentences:
            matcher = SequenceMatcher(None, orig_sent, src_sent)
            blocks = matcher.get_matching_blocks()
            
            for block in blocks:
                if block.size >= min_length:
                    matches.append({
                        "original": orig_sent[block.a:block.a + block.size],
                        "source": src_sent[block.b:block.b + block.size],
                        "size": block.size
                    })
    
    # Remove duplicates
    unique_matches = []
    seen = set()
    for match in matches:
        if match["original"] not in seen and len(match["original"].split()) >= 3:
            seen.add(match["original"])
            unique_matches.append(match)
    
    return unique_matches

def check_plagiarism(text):
    """
    Check plagiarism by:
    1. Breaking text into smaller chunks
    2. Searching each chunk online
    3. Comparing the text with search results
    4. Identifying matching content
    5. Performing semantic analysis to detect paraphrased content
    """
    logger.debug("Starting plagiarism check")
    
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Extract sentences for search queries
    sentences = sent_tokenize(text)
    
    # Take every third sentence to create search queries
    search_queries = [sentences[i] for i in range(0, len(sentences), 3)]
    
    # Limit to max 5 queries to avoid API overuse
    search_queries = search_queries[:5]
    
    results = []
    source_texts = []  # Store source texts for semantic comparison
    
    # For each query, search online and compare results
    for query in search_queries:
        # Search online with increased number of results
        search_results = search_online(query, num_results=20)
        
        for result in search_results:
            try:
                # Get content from the webpage
                source_url = result["link"]
                if source_url == "#":  # Skip placeholder URLs
                    continue
                    
                source_content = get_website_text_content(source_url)
                
                if not source_content:
                    continue
                
                # Store source text for semantic comparison
                source_texts.append({
                    "content": source_content,
                    "title": result["title"],
                    "url": source_url,
                    "snippet": result["snippet"]
                })
                
                # Skip empty content
                if not source_content or len(source_content.strip()) < 50:
                    logger.warning(f"Source content too short or empty from {source_url}")
                    continue
                
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
                        results.append({
                            "source": {
                                "title": result["title"],
                                "url": source_url,
                                "snippet": result["snippet"]
                            },
                            "similarity": similarity_percent,
                            "matches": matches,
                            "semantic_matches": []  # Will be filled later
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
    
    # Perform semantic comparison if we have sources to compare
    if source_texts:
        try:
            logger.info(f"Performing semantic comparison with {len(source_texts)} sources")
            semantic_results = semantic_comparison.check_semantic_plagiarism(text, source_texts)
            
            # Merge lexical and semantic results
            merged_results = []
            
            # Track seen URLs to avoid duplicates
            semantic_urls = {r["source"]["url"]: r for r in semantic_results}
            lexical_urls = {r["source"]["url"]: r for r in results}
            
            # Process all unique URLs
            all_urls = set(list(semantic_urls.keys()) + list(lexical_urls.keys()))
            
            for url in all_urls:
                if url in lexical_urls and url in semantic_urls:
                    # Merge the two results
                    lex_result = lexical_urls[url]
                    sem_result = semantic_urls[url]
                    
                    # Use the higher similarity score
                    if sem_result["similarity"] > lex_result["similarity"]:
                        lex_result["similarity"] = sem_result["similarity"]
                    
                    # Add semantic matches
                    lex_result["semantic_matches"] = sem_result.get("semantic_matches", [])
                    
                    # Combine matches, avoiding duplicates by using text as key
                    existing_matches = {m["original"]: m for m in lex_result["matches"]}
                    for match in sem_result.get("matches", []):
                        if match["original"] not in existing_matches:
                            lex_result["matches"].append(match)
                    
                    merged_results.append(lex_result)
                elif url in lexical_urls:
                    merged_results.append(lexical_urls[url])
                else:
                    merged_results.append(semantic_urls[url])
            
            # Replace results with merged results
            results = merged_results
            
        except Exception as e:
            logger.error(f"Error in semantic comparison: {str(e)}")
    
    # Sort results by similarity (descending)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
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
            "match_count": len(result["matches"])
        }
        top_sources.append(source_info)
    
    # Create report data
    report = {
        "original_length": len(original_text.split()),
        "sources_checked": len(results),
        "average_similarity": avg_similarity,
        "total_matches": total_matches,
        "top_sources": top_sources,
        "timestamp": import_datetime().datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return report

def import_datetime():
    """Import datetime module on demand to avoid circular imports."""
    import datetime
    return datetime
