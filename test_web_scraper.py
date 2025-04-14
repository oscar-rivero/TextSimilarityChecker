import logging
from web_scraper import get_website_text_content, get_mock_content

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_web_scraper():
    """Test the web scraper functionality with safe mode enabled."""
    test_urls = [
        "https://example.com/page1",
        "https://testdomain.org/article/123",
        "https://university.edu/research/ai",
        "https://invalid.url",
        "https://blog.website.com/post/plagiarism-detection"
    ]
    
    for url in test_urls:
        logger.info(f"Testing URL: {url}")
        content = get_website_text_content(url)
        logger.info(f"Content length: {len(content)} characters")
        # Print a sample of the content
        logger.info(f"Content sample: {content[:100]}...")
        logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting web scraper test")
    test_web_scraper()
    logger.info("Web scraper test completed")