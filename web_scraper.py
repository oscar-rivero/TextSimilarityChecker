import trafilatura
import logging
import requests
from urllib.parse import urlparse
import time
import http.client

# Set shorter timeout for HTTP connections to prevent hanging
http.client._MAXHEADERS = 100
requests.packages.urllib3.util.connection.TIMEOUT_DEFAULT = 10

logger = logging.getLogger(__name__)

def is_valid_url(url):
    """Check if URL is valid and has a supported scheme"""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    """
    # Check if URL is valid first
    if not is_valid_url(url):
        logger.warning(f"Invalid URL format: {url}")
        return ""
        
    # Add user agent to avoid being blocked
    custom_config = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'TIMEOUT': 10,  # 10 seconds timeout
    }
    
    try:
        # Send a request to the website with timeout
        start_time = time.time()
        downloaded = trafilatura.fetch_url(url, config=custom_config)
        
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
