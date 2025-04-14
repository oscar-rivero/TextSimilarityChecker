import trafilatura
import logging
import requests
from urllib.parse import urlparse
import time
import os
import hashlib
import random

logger = logging.getLogger(__name__)

# Set to True to use mock content instead of real web requests
USE_SAFE_MODE = True

def is_valid_url(url):
    """Check if URL is valid and has a supported scheme"""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def get_mock_content(url):
    """Generate deterministic mock content based on URL"""
    # Create a hash of the URL to ensure consistent results for the same URL
    hash_obj = hashlib.md5(url.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Use the hash to seed the random generator for consistent results
    random.seed(int(hash_hex, 16) % (2**32))
    
    # Create a set of random paragraphs based on the URL
    domain = urlparse(url).netloc
    paragraphs = [
        f"This is mock content for {domain}.",
        f"The URL {url} contains information about various topics.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Phasellus at dolor vel eros consectetur condimentum.",
        "Mauris semper justo ut sapien feugiat, at finibus metus molestie.",
        "Proin sit amet nisl euismod, ultrices lectus vel, maximus enim.",
    ]
    
    # Select a random number of paragraphs between 2 and 5
    num_paragraphs = random.randint(2, 5)
    selected_paragraphs = random.sample(paragraphs, num_paragraphs)
    
    # Join with newlines
    mock_content = "\n\n".join(selected_paragraphs)
    return mock_content

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    In safe mode, it returns mock content to prevent errors.
    """
    # Check if URL is valid first
    if not is_valid_url(url):
        logger.warning(f"Invalid URL format: {url}")
        return ""
    
    # Use mock content in safe mode to prevent errors
    if USE_SAFE_MODE:
        logger.info(f"Using mock content for {url} (safe mode)")
        return get_mock_content(url)
    
    try:
        # Simple approach without custom config - trafilatura handles user agent and timeouts internally
        start_time = time.time()
        downloaded = trafilatura.fetch_url(url)
        
        # If it takes too long, bail out
        if time.time() - start_time > 15:  # 15 seconds maximum
            logger.warning(f"Timeout fetching content from {url}")
            return ""
            
        if not downloaded:
            logger.warning(f"Failed to download content from {url}")
            return ""
            
        text = trafilatura.extract(downloaded)
        
        if not text:
            logger.warning(f"Failed to extract text from {url}")
            return ""
            
        return text
    except requests.exceptions.Timeout:
        logger.warning(f"Request timeout for {url}")
        return ""
    except requests.exceptions.TooManyRedirects:
        logger.warning(f"Too many redirects for {url}")
        return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return ""
