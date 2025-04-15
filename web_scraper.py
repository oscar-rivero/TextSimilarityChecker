import trafilatura
import logging
import requests
from urllib.parse import urlparse
import time
import os
import hashlib
import random

logger = logging.getLogger(__name__)

# Set to False to fetch real webpage content instead of using mock data
# This will fetch and analyze actual HTML content from websites
USE_SAFE_MODE = False

def is_valid_url(url):
    """Check if URL is valid and has a supported scheme"""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def get_mock_content(url):
    """Generate realistic mock content based on URL for testing"""
    # Create a hash of the URL to ensure consistent results for the same URL
    hash_obj = hashlib.md5(url.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Use the hash to seed the random generator for consistent results
    random.seed(int(hash_hex, 16) % (2**32))
    
    # Extract domain and path from URL
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # Handle Wikipedia URLs with realistic content
    if "wikipedia" in url.lower() or "en.wiki" in url.lower():
        # Extract entity from the URL
        parts = path.split('/')
        entity = parts[-1] if parts and len(parts) > 1 else "Unknown"
        entity = entity.replace('_', ' ').replace('%20', ' ')
        
        # Categories and their content templates
        content_templates = {
            'history': [
                f"{entity} has been an important historical subject since ancient times. Historical records show that {entity} played a crucial role in the development of several civilizations.",
                f"In the medieval period, {entity} was documented by several European chroniclers who noted its significance in religious and cultural contexts.",
                f"During the Renaissance, scholars began to study {entity} more systematically, applying new methods of historical inquiry and analysis.",
                f"The modern understanding of {entity} has been shaped by archaeological discoveries in the 20th century that revealed new dimensions of its historical importance.",
                f"Historians continue to debate various aspects of {entity}, particularly its influence on contemporary cultural and political developments."
            ],
            'science': [
                f"{entity} is a significant concept in scientific research. Scientists have conducted numerous studies to understand its fundamental properties and behavior.",
                f"Research on {entity} has led to several breakthroughs in understanding natural phenomena and has applications in various fields including medicine and technology.",
                f"The chemical and physical properties of {entity} make it particularly valuable for scientific analysis and experimental studies.",
                f"Recent advances in technology have allowed researchers to examine {entity} at unprecedented levels of detail, revealing complex structures and interactions.",
                f"The scientific community continues to explore the implications of {entity} for theoretical frameworks in physics, biology, and chemistry."
            ],
            'literature': [
                f"{entity} represents an important theme in literary works throughout history. Authors have used it as a metaphor for human experience and cultural values.",
                f"Literary critics have analyzed the symbolism of {entity} in various texts, noting how it reflects broader social and philosophical concerns.",
                f"In poetry, {entity} often serves as a powerful image that evokes emotional responses and philosophical reflection.",
                f"The narrative treatment of {entity} varies across different literary traditions, reflecting diverse cultural perspectives and aesthetic values.",
                f"Contemporary literature continues to explore {entity} in new ways, often challenging traditional interpretations and introducing innovative narrative strategies."
            ],
            'technology': [
                f"{entity} has significantly influenced technological development in recent decades. Engineers have incorporated its principles into various systems and devices.",
                f"The technological applications of {entity} span multiple industries, from communications to manufacturing and healthcare.",
                f"Recent innovations in {entity} technology have led to improved efficiency and new capabilities in electronic systems.",
                f"Researchers continue to explore how {entity} can be integrated with artificial intelligence and machine learning algorithms.",
                f"The future development of {entity} technology will likely focus on sustainability and addressing global challenges."
            ],
            'biology': [
                f"{entity} plays a crucial role in biological systems. It is involved in essential processes that maintain cellular function and organism health.",
                f"Biologists have studied how {entity} interacts with various cellular components and contributes to physiological regulation.",
                f"The evolutionary development of {entity} provides insights into adaptation mechanisms and species diversification.",
                f"In ecological contexts, {entity} influences interactions between organisms and their environments, contributing to ecosystem stability.",
                f"Current research on {entity} includes investigations of its potential applications in biotechnology and medicine."
            ]
        }
        
        # Select category based on URL and hash
        categories = list(content_templates.keys())
        category_index = int(hash_hex[0], 16) % len(categories)
        selected_category = categories[category_index]
        
        # Create title and introduction
        title = f"{entity.title()}"
        introduction = f"{entity.title()} is a topic of considerable interest in the field of {selected_category}. The following information provides an overview of key aspects and recent developments."
        
        # Select paragraphs from the appropriate category
        topic_paragraphs = content_templates[selected_category]
        num_paragraphs = min(4, len(topic_paragraphs))
        selected_paragraphs = random.sample(topic_paragraphs, num_paragraphs)
        
        # Special handling for common topics
        if "History_of_Ethiopia" in url or "Ethiopian_history" in url:
            special_content = [
                "The history of Ethiopian historiography has been dominated traditionally by theology of Christianity and the chronology of the Bible.",
                "The literature of Ethiopian historiography has been dominated traditionally by theology of Christianity and the chronology of the Bible.",
                "Ethiopian history was recorded predominantly by monks and court historians, often blending historical events with theological interpretation.",
                "Medieval European chroniclers often referenced Ethiopia in their writings, connecting it to religious narratives and the mythical kingdom of Prester John."
            ]
            selected_paragraphs = special_content + selected_paragraphs
        
        # Assemble the content with title and introduction
        full_content = [title, introduction] + selected_paragraphs
        return "\n\n".join(full_content)
    
    # For search engine results pages, provide a directory-like listing
    elif "search" in url.lower() or "index.php" in url.lower() or "opensearch" in url.lower():
        search_term = path.split('=')[-1].replace('+', ' ').replace('%20', ' ')
        results_intro = f"Search results for '{search_term}'. The following resources may contain relevant information:"
        
        # Generate mock search results
        results = [
            f"1. Introduction to {search_term} - A comprehensive overview of key concepts and historical development.",
            f"2. {search_term} in Contemporary Research - Recent studies and scientific advances in this field.",
            f"3. The Role of {search_term} in Society - Analysis of cultural, economic, and social impacts.",
            f"4. {search_term}: A Historical Perspective - How understanding has evolved over time.",
            f"5. Future Directions in {search_term} Studies - Emerging trends and potential developments."
        ]
        
        full_content = [results_intro] + results
        return "\n\n".join(full_content)
    
    # For all other domains, generate domain-appropriate content
    else:
        domain_name = domain.split('.')[-2] if len(domain.split('.')) > 1 else domain
        title = f"Information about {domain_name.title()}"
        introduction = f"This website provides information about various topics related to {domain_name}. The content below represents key areas covered on this site."
        
        # Generate domain-specific paragraphs
        domain_paragraphs = [
            f"The field of {domain_name} encompasses a wide range of concepts, methodologies, and applications that continue to evolve with new research and technological developments.",
            f"Practitioners in {domain_name} apply specialized knowledge to address complex problems and develop innovative solutions that benefit various sectors.",
            f"Historical developments in {domain_name} show a progression from basic principles to sophisticated frameworks that integrate multiple disciplinary perspectives.",
            f"Current challenges in {domain_name} include adapting to technological change, addressing ethical considerations, and ensuring sustainable practices.",
            f"The future of {domain_name} will likely be shaped by emerging technologies, changing societal needs, and new approaches to traditional problems."
        ]
        
        # Select a subset of paragraphs
        num_paragraphs = random.randint(3, 5)
        selected_paragraphs = random.sample(domain_paragraphs, num_paragraphs)
        
        # Assemble the content
        full_content = [title, introduction] + selected_paragraphs
        return "\n\n".join(full_content)

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura which handles proper text extraction.
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
        logger.info(f"Fetching real content from {url}")
        # Configure trafilatura for better extraction
        start_time = time.time()
        
        # Set custom headers to simulate a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Use requests for more control over the request
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # If it takes too long, bail out
        if time.time() - start_time > 15:  # 15 seconds maximum
            logger.warning(f"Timeout fetching content from {url}")
            return ""
            
        # Use trafilatura to extract the text from the HTML
        text = trafilatura.extract(response.text)
        
        if not text or len(text.strip()) < 50:
            # Fallback to BeautifulSoup if trafilatura fails to extract content
            logger.info(f"Trafilatura failed to extract content, using BeautifulSoup as fallback for {url}")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
                
            # Get the text content
            text = soup.get_text(separator='\n\n')
            
            # Clean the text (remove extra whitespace, etc.)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n\n'.join(lines)
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"Failed to extract meaningful text from {url}")
            return ""
            
        logger.info(f"Successfully extracted {len(text)} characters from {url}")
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
