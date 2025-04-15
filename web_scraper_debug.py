import os
import sys
import logging
import traceback
import datetime
import requests
from urllib.parse import urlparse
import re

# Configuración de logging para debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='web_scraper_error.log',
    filemode='a'
)

logger = logging.getLogger('web_scraper_debug')

def debug_scraper(url):
    """
    Función para depurar el scraper con un URL específico
    Registra todo el proceso y cualquier error en el archivo de log
    """
    logger.info("="*80)
    logger.info(f"DEBUG SESSION START: {datetime.datetime.now().isoformat()}")
    logger.info(f"Testing URL: {url}")
    logger.info("="*80)
    
    try:
        # 1. Validar URL
        if not url or not isinstance(url, str):
            logger.warning(f"Invalid URL: {url} (not a string or empty)")
            return "INVALID URL"
            
        # 2. Corregir URL si es necesario
        if not url.startswith('http://') and not url.startswith('https://'):
            original_url = url
            url = 'https://' + url
            logger.info(f"URL corrected from {original_url} to {url}")
        
        # 3. Validar URL después de corrección
        try:
            result = urlparse(url)
            valid_url = all([result.scheme in ['http', 'https'], result.netloc])
            logger.info(f"URL validation result: {valid_url}")
            logger.info(f"Parsed URL: scheme={result.scheme}, netloc={result.netloc}, path={result.path}")
            
            if not valid_url:
                return "URL INVALID AFTER CORRECTION"
        except Exception as parse_error:
            logger.error(f"URL parsing error: {str(parse_error)}")
            logger.error(traceback.format_exc())
            return "URL PARSE ERROR"
        
        # 4. Enviar solicitud HTTP
        logger.info(f"Sending HTTP request to {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            logger.info("Starting HTTP request")
            response = requests.get(url, headers=headers, timeout=10, 
                               allow_redirects=True, stream=True)
            logger.info(f"Request status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            # Verificar si la respuesta fue exitosa
            response.raise_for_status()
        except Exception as request_error:
            logger.error(f"HTTP request error: {str(request_error)}")
            logger.error(traceback.format_exc())
            return "HTTP REQUEST ERROR"
        
        # 5. Procesar el contenido HTML
        try:
            # Limitar tamaño del HTML
            content_length = response.headers.get('Content-Length')
            logger.info(f"Content length: {content_length}")
            
            if content_length and int(content_length) > 500*1024:
                logger.warning(f"Content too large: {content_length} bytes, truncating to 500KB")
                html_content = response.raw.read(500*1024).decode('utf-8', errors='ignore')
            else:
                html_content = response.text[:500*1024] if len(response.text) > 500*1024 else response.text
            
            logger.info(f"HTML content size: {len(html_content)} bytes")
            logger.info(f"HTML content sample (first 200 chars): {html_content[:200]}")
        except Exception as content_error:
            logger.error(f"HTML content processing error: {str(content_error)}")
            logger.error(traceback.format_exc())
            return "HTML CONTENT ERROR"
        
        # 6. Limpiar HTML - Fase 1: Eliminar scripts, styles, etc.
        try:
            logger.info("Starting HTML cleaning phase 1")
            cleaned_html = re.sub(r'<(script|style|head)\b[^<]*(?:(?!</(script|style|head)>)<[^<]*)*</(script|style|head)>', 
                                '', html_content, flags=re.IGNORECASE | re.DOTALL)
            logger.info(f"HTML cleaning phase 1 complete. New size: {len(cleaned_html)} bytes")
        except Exception as cleaning_error:
            logger.error(f"HTML cleaning phase 1 error: {str(cleaning_error)}")
            logger.error(traceback.format_exc())
            return "HTML CLEANING ERROR"
        
        # 7. Extraer elementos de contenido específicos
        try:
            logger.info("Starting content element extraction")
            content_elements = re.findall(r'<(article|main|div id="content"|div class="content").*?</\1>', 
                                      cleaned_html, re.DOTALL | re.IGNORECASE)
            logger.info(f"Found {len(content_elements)} content elements")
            
            if content_elements:
                logger.info(f"First content element sample: {content_elements[0][:200]}")
        except Exception as element_error:
            logger.error(f"Content element extraction error: {str(element_error)}")
            logger.error(traceback.format_exc())
            logger.info("Continuing with empty content elements list")
            content_elements = []
        
        # 8. Procesar elementos de contenido
        text = ""
        if content_elements:
            try:
                logger.info("Processing content elements")
                element_texts = []
                for i, element in enumerate(content_elements[:5]):  # Procesar solo los primeros 5
                    try:
                        # Eliminar etiquetas HTML
                        element_text = re.sub(r'<[^>]+>', ' ', element)
                        element_text = re.sub(r'\s+', ' ', element_text).strip()
                        
                        logger.info(f"Element {i} processed. Size: {len(element_text)} chars")
                        
                        if len(element_text) > 100:
                            element_texts.append(element_text)
                            logger.info(f"Element {i} added to results (sufficient size)")
                        else:
                            logger.info(f"Element {i} ignored (insufficient size)")
                    except Exception as element_process_error:
                        logger.error(f"Error processing element {i}: {str(element_process_error)}")
                
                if element_texts:
                    text = "\n\n".join(element_texts)
                    logger.info(f"Successfully extracted content from elements. Total size: {len(text)} chars")
            except Exception as elements_error:
                logger.error(f"Error processing content elements: {str(elements_error)}")
                logger.error(traceback.format_exc())
        
        # 9. Procesar documento completo si no se obtuvo contenido útil
        if not text or len(text) < 200:
            try:
                logger.info("No useful content from elements, processing whole document")
                
                # Eliminar todas las etiquetas HTML
                text = re.sub(r'<[^>]+>', ' ', cleaned_html)
                logger.info(f"HTML tags removed. New size: {len(text)} chars")
                
                # Limpiar espacios en blanco
                text = re.sub(r'\s+', ' ', text).strip()
                logger.info(f"Whitespace cleaned. New size: {len(text)} chars")
                
                # Dividir en párrafos
                paragraphs = re.split(r'\s{2,}', text)
                logger.info(f"Split into {len(paragraphs)} potential paragraphs")
                
                # Filtrar párrafos significativos
                paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
                logger.info(f"Filtered to {len(paragraphs)} significant paragraphs")
                
                # Unir párrafos significativos
                text = "\n\n".join(paragraphs)
                logger.info(f"Final text size from whole document: {len(text)} chars")
            except Exception as whole_doc_error:
                logger.error(f"Error processing whole document: {str(whole_doc_error)}")
                logger.error(traceback.format_exc())
                
                # Fallback extremo
                try:
                    logger.info("Attempting extreme fallback processing")
                    text = re.sub(r'<[^>]*>', ' ', html_content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    logger.info(f"Extreme fallback result: {len(text)} chars")
                except Exception as extreme_fallback_error:
                    logger.error(f"Even extreme fallback failed: {str(extreme_fallback_error)}")
                    text = html_content.replace('<', ' ').replace('>', ' ')
                    text = re.sub(r'\s+', ' ', text).strip()
                    logger.info(f"Ultimate fallback result: {len(text)} chars")
        
        # 10. Verificar el contenido final
        if not text or len(text.strip()) < 100:
            logger.warning(f"Failed to extract meaningful text from {url}")
            return "NO MEANINGFUL CONTENT"
        
        # 11. Limitar la longitud final del texto
        if len(text) > 50000:
            logger.info(f"Limiting extracted text to 50000 chars (was {len(text)})")
            text = text[:50000]
        
        logger.info(f"Successfully extracted {len(text)} characters from {url}")
        logger.info("Text sample (first 200 chars): " + text[:200])
        logger.info("="*80)
        logger.info(f"DEBUG SESSION END: {datetime.datetime.now().isoformat()}")
        logger.info("="*80)
        
        return f"SUCCESS: {len(text)} chars extracted"
    
    except Exception as e:
        logger.error(f"Unhandled exception in debug_scraper: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("="*80)
        logger.info(f"DEBUG SESSION END WITH ERROR: {datetime.datetime.now().isoformat()}")
        logger.info("="*80)
        return f"UNHANDLED ERROR: {str(e)}"

# Si este script se ejecuta directamente, probar con algunas URLs
if __name__ == "__main__":
    test_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://www.bbc.com/news",
        "https://www.nytimes.com",
        "https://www.sciencedirect.com/science/article/pii/S0048969720359209"
    ]
    
    if len(sys.argv) > 1:
        # Si se proporciona una URL específica en la línea de comandos
        test_urls = [sys.argv[1]]
    
    print("Web Scraper Debugging Tool")
    print("=" * 50)
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        result = debug_scraper(url)
        print(f"Result: {result}")
        print("-" * 50)
    
    print(f"\nDebug log written to: {os.path.abspath('web_scraper_error.log')}")