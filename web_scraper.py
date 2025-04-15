import logging
import requests
import re
from urllib.parse import urlparse
import time
import os
import hashlib
import random

logger = logging.getLogger(__name__)

# Desactivamos el modo seguro para usar datos reales como ha solicitado el usuario
# Implementamos manejo robusto de errores para evitar problemas
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
    The text content is extracted using the trafilatura library for optimal content extraction.
    In safe mode, it returns mock content to prevent errors.
    """
    # Check if URL is valid first
    if not url or not isinstance(url, str):
        logger.warning(f"Invalid URL: {url} (not a string or empty)")
        return ""
        
    # Fix common URL issues
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    if not is_valid_url(url):
        logger.warning(f"Invalid URL format after fixing: {url}")
        return ""
    
    # Use mock content in safe mode to prevent errors
    if USE_SAFE_MODE:
        logger.info(f"Using mock content for {url} (safe mode)")
        return get_mock_content(url)
    
    try:
        logger.info(f"Fetching real content from {url}")
        
        # Desactivamos temporalmente trafilatura debido a problemas
        # y usamos directamente nuestro método alternativo más robusto
        logger.info(f"Using robust alternative method for {url}")
        
        # Método alternativo con requests y procesamiento básico
        # Set custom headers to simulate a modern browser with better compatibility
        # Especialmente diseñado para sitios académicos y editoriales
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'sec-ch-ua': '"Google Chrome";v="121", "Not A(Brand";v="24", "Chromium";v="121"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'cross-site',
            'sec-fetch-user': '?1',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        # Petición con manejo de errores y tiempo de espera reducido
        # Implementamos manejo especial para sitios académicos y editoriales
        try:
            # Primero intentamos con la configuración normal
            response = requests.get(url, headers=headers, timeout=10, 
                                   allow_redirects=True, stream=True)
            
            # Verificamos si nos redirigió a una página de "unsupported browser"
            if 'unsupported' in response.url.lower() or response.status_code == 400:
                logger.warning(f"Detected unsupported browser page: {response.url}. Trying alternative approach")
                
                # Modificamos los headers para simular mejor un navegador real
                alt_headers = headers.copy()
                alt_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0'
                alt_headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                
                # Intentamos de nuevo con la URL original
                logger.info(f"Retrying with alternative headers: {url}")
                original_url = url
                session = requests.Session()
                
                # Configuración manual para seguir redirecciones
                response = session.get(url, headers=alt_headers, timeout=15, stream=True, allow_redirects=False)
                
                # Seguimos manualmente hasta 10 redirecciones
                redirect_count = 0
                while response.is_redirect and redirect_count < 10:
                    redirect_url = response.headers['Location']
                    # Convertir URLs relativas a absolutas
                    if redirect_url.startswith('/'):
                        parsed_url = urlparse(url)
                        redirect_url = f"{parsed_url.scheme}://{parsed_url.netloc}{redirect_url}"
                    logger.info(f"Following redirect: {redirect_url}")
                    response = session.get(redirect_url, headers=alt_headers, timeout=15, stream=True, allow_redirects=False)
                    redirect_count += 1
                
                if redirect_count >= 10:
                    logger.error(f"Too many redirects for {url}")
                    return ""
            
            # Verificamos si la respuesta fue exitosa después de todo el manejo
            response.raise_for_status()
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error for {url}: {str(http_err)}")
            # Obtener el código de estado de la excepción en lugar de la variable response
            status_code = getattr(http_err.response, 'status_code', 0)
            if status_code in [403, 429]:
                logger.warning(f"Access denied (status code {status_code}). Site may be blocking scrapers.")
            return ""
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error for {url}: {str(conn_err)}")
            return ""
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error for {url}: {str(timeout_err)}")
            return ""
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error for {url}: {str(req_err)}")
            return ""
        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {str(e)}")
            return ""
        
        # Extraemos el contenido con límite de tamaño
        try:
            # Limit HTML size to prevent memory issues (500KB max)
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > 500*1024:
                logger.warning(f"Content too large: {content_length} bytes, truncating to 500KB")
                html_content = response.raw.read(500*1024).decode('utf-8', errors='ignore')
            else:
                html_content = response.text[:500*1024] if len(response.text) > 500*1024 else response.text
            
            # HTML cleaning process with fallbacks
            text = ""
            
            # First attempt: Remove scripts, styles, and HTML tags
            try:
                # Remove script, style and head tags
                cleaned_html = re.sub(r'<(script|style|head)\b[^<]*(?:(?!</(script|style|head)>)<[^<]*)*</(script|style|head)>', 
                                    '', html_content, flags=re.IGNORECASE | re.DOTALL)
                
                # Extract text from specific content elements first (more likely to contain useful text)
                try:
                    content_elements = re.findall(r'<(article|main|div id="content"|div class="content").*?</\1>', 
                                              cleaned_html, re.DOTALL | re.IGNORECASE)
                except Exception as element_error:
                    logger.error(f"Error finding content elements: {str(element_error)}")
                    content_elements = []
                
                if content_elements:
                    # Process each content element
                    element_texts = []
                    for element in content_elements:
                        # Remove HTML tags and clean up
                        element_text = re.sub(r'<[^>]+>', ' ', element)
                        element_text = re.sub(r'\s+', ' ', element_text).strip()
                        if len(element_text) > 100:  # Only include significant text blocks
                            element_texts.append(element_text)
                    
                    if element_texts:
                        text = "\n\n".join(element_texts)
                
                # If we didn't get good content from content elements, process whole document
                if not text or len(text) < 200:
                    # Remove all HTML tags
                    text = re.sub(r'<[^>]+>', ' ', cleaned_html)
                    
                    # Clean up whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Split into paragraphs (text blocks separated by multiple spaces)
                    paragraphs = re.split(r'\s{2,}', text)
                    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
                    
                    # Join significant paragraphs
                    text = "\n\n".join(paragraphs)
            except Exception as regex_error:
                logger.error(f"Primary HTML processing failed: {str(regex_error)}")
                # Fallback to very basic extraction
                try:
                    # Strip HTML tags brutally
                    text = re.sub(r'<[^>]*>', ' ', html_content)
                    text = re.sub(r'\s+', ' ', text).strip()
                except Exception as e:
                    logger.error(f"Even basic HTML processing failed: {str(e)}")
                    # Ultimate fallback - just remove brackets
                    text = html_content.replace('<', ' ').replace('>', ' ')
                    text = re.sub(r'\s+', ' ', text).strip()
            
            # Detectar si es un sitio que probablemente requiere JavaScript
            js_required_indicators = [
                "Please enable JavaScript",
                "JavaScript is required",
                "enable JavaScript to continue",
                "This page requires JavaScript",
                "Your browser has JavaScript disabled",
                "You need to enable JavaScript",
                "content is not available without JavaScript"
            ]
            
            js_required = False
            for indicator in js_required_indicators:
                if indicator.lower() in html_content.lower():
                    js_required = True
                    logger.warning(f"Detected JavaScript requirement on {url}")
                    break
            
            # Verify we have meaningful content
            if not text or len(text.strip()) < 100 or js_required:
                logger.warning(f"Failed to extract meaningful text from {url} - JS required: {js_required}")
                
                # Para sitios que requieren JavaScript o donde no se pudo extraer contenido,
                # podemos hacer una búsqueda de metadatos como fallback
                try:
                    # Extraer metadatos de las etiquetas meta
                    meta_description = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
                    meta_keywords = re.search(r'<meta\s+name=["\']keywords["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
                    og_description = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
                    og_title = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
                    
                    # Extraer título
                    title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
                    
                    # Combinar metadatos disponibles para crear un resumen
                    meta_parts = []
                    if title_match:
                        meta_parts.append(f"Title: {title_match.group(1)}")
                    if meta_description:
                        meta_parts.append(f"Description: {meta_description.group(1)}")
                    if og_description:
                        meta_parts.append(f"Social description: {og_description.group(1)}")
                    if og_title and not title_match:
                        meta_parts.append(f"Social title: {og_title.group(1)}")
                    if meta_keywords:
                        meta_parts.append(f"Keywords: {meta_keywords.group(1)}")
                    
                    # Si tenemos algún metadato, usarlo como texto
                    if meta_parts:
                        meta_text = "\n\n".join(meta_parts)
                        logger.info(f"Using metadata as fallback content for {url}, length: {len(meta_text)}")
                        if len(meta_text) > 100:
                            return meta_text
                except Exception as meta_error:
                    logger.error(f"Failed to extract metadata from {url}: {str(meta_error)}")
                
                # Si llegamos aquí, no pudimos extraer nada útil
                return ""
            
            # Limit final text length
            if len(text) > 50000:
                logger.info(f"Limiting extracted text to 50000 chars (was {len(text)})")
                text = text[:50000]
                
            logger.info(f"Successfully extracted {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return ""
            
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
