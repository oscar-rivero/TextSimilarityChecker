import trafilatura
import logging

logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            logger.warning(f"Failed to download content from {url}")
            return ""
            
        text = trafilatura.extract(downloaded)
        
        if not text:
            logger.warning(f"Failed to extract text from {url}")
            return ""
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return ""
