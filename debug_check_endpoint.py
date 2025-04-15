import requests
import json
import logging
import traceback
import os
import sys
import time
from urllib.parse import urljoin
import re

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='endpoint_debug.log',
    filemode='a'
)

logger = logging.getLogger('endpoint_debug')

# URL base de la aplicación
BASE_URL = "http://localhost:5000"
ENDPOINT = "/check/"  # Importante: agregamos slash al final

# Textos de prueba para diferentes escenarios
TEST_TEXTS = [
    {
        "name": "Texto corto genérico",
        "text": "Este es un texto de prueba para el detector de plagio."
    },
    {
        "name": "Texto sobre historia",
        "text": "La historia de Egipto es una de las más antiguas del mundo. La civilización se desarrolló en el valle del Nilo hace más de 5000 años y dejó un legado cultural y arquitectónico impresionante, incluyendo las pirámides y la Esfinge."
    },
    {
        "name": "Texto sobre tecnología",
        "text": "La inteligencia artificial está revolucionando muchos campos. Los algoritmos de aprendizaje profundo pueden analizar grandes cantidades de datos y encontrar patrones que los humanos podrían pasar por alto."
    },
    {
        "name": "Texto sobre medicina",
        "text": "El sistema circulatorio humano transporta oxígeno y nutrientes a todas las células del cuerpo. Está compuesto por el corazón, vasos sanguíneos y aproximadamente 5 litros de sangre."
    }
]

def check_endpoint(text):
    """
    Función para probar el endpoint /check con un texto dado
    """
    logger.info("="*80)
    logger.info(f"INICIANDO PRUEBA: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Texto de prueba: {text[:100]}...")
    logger.info("="*80)
    
    try:
        # Primero necesitamos obtener una cookie CSRF válida
        session = requests.Session()
        
        # 1. Obtenemos la página principal para obtener una cookie CSRF
        logger.debug("Obteniendo página principal para cookie CSRF")
        index_response = session.get(BASE_URL)
        
        if index_response.status_code != 200:
            logger.error(f"Error al obtener página principal: {index_response.status_code}")
            return False, f"Error al obtener página principal: {index_response.status_code}"
        
        # Buscar el token CSRF en la página
        csrf_token = None
        csrf_match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', index_response.text)
        if csrf_match:
            csrf_token = csrf_match.group(1)
            logger.debug(f"Token CSRF encontrado: {csrf_token[:8]}...")
        else:
            logger.warning("No se encontró token CSRF en la página. Intentando con cookies solamente.")
        
        # Preparar la solicitud
        url = urljoin(BASE_URL, ENDPOINT)
        data = {"text": text}
        
        # Añadir token CSRF si está disponible
        if csrf_token:
            data["csrfmiddlewaretoken"] = csrf_token
        
        # Preparar los headers necesarios para CSRF
        headers = {
            "Referer": BASE_URL,
            "X-Requested-With": "XMLHttpRequest"
        }
        
        logger.debug(f"Enviando solicitud POST a {url}")
        logger.debug(f"Datos de la solicitud: {data}")
        
        # Enviar la solicitud
        start_time = time.time()
        response = session.post(url, data=data, headers=headers)
        end_time = time.time()
        
        # Registrar información básica de la respuesta
        logger.info(f"Código de estado: {response.status_code}")
        logger.info(f"Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        # Analizar la respuesta
        if response.status_code == 200:
            logger.info("Respuesta exitosa (200 OK)")
            
            # Comprobar si hay resultados
            if "No se encontraron resultados" in response.text:
                logger.warning("No se encontraron resultados de plagio")
                return False, "No se encontraron resultados"
            
            # Buscar el número de resultados en la respuesta HTML
            results_match = re.search(r'Resultados encontrados: (\d+)', response.text)
            if results_match:
                num_results = int(results_match.group(1))
                logger.info(f"Número de resultados encontrados: {num_results}")
                
                if num_results > 0:
                    logger.info("Éxito: Se encontraron resultados")
                    
                    # Buscar los porcentajes de similitud en la respuesta HTML
                    similarity_matches = re.findall(r'(\d+)% de coincidencia', response.text)
                    if similarity_matches:
                        logger.info(f"Porcentajes de similitud encontrados: {similarity_matches}")
                    
                    return True, f"Éxito: {num_results} resultados"
                else:
                    logger.warning("No hay resultados (contador es 0)")
                    return False, "No hay resultados (contador es 0)"
            else:
                logger.warning("No se pudo encontrar el contador de resultados en la respuesta")
                return False, "No se pudo encontrar el contador de resultados"
                
        elif response.status_code == 500:
            logger.error("Error Interno del Servidor (500)")
            logger.error(f"Contenido de la respuesta: {response.text[:500]}")
            
            # Comprobar si hay un mensaje de error específico
            error_match = re.search(r'<p>(.+?)</p>', response.text)
            if error_match:
                error_message = error_match.group(1)
                logger.error(f"Mensaje de error: {error_message}")
                return False, f"Error 500: {error_message}"
            else:
                return False, "Error 500 sin mensaje específico"
        else:
            logger.error(f"Código de estado inesperado: {response.status_code}")
            logger.error(f"Contenido de la respuesta: {response.text[:500]}")
            return False, f"Código de estado inesperado: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la solicitud HTTP: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Error en la solicitud HTTP: {str(e)}"
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Error inesperado: {str(e)}"
    finally:
        logger.info("="*80)
        logger.info(f"FIN DE PRUEBA: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

def check_web_scraper_log():
    """
    Función para verificar el archivo web_scraper_error.log
    """
    try:
        if os.path.exists('web_scraper_error.log'):
            with open('web_scraper_error.log', 'r') as f:
                log_content = f.read()
                
            # Buscar errores comunes
            error_patterns = [
                ('Timeout', r'timeout|timed? out'),
                ('HTTP Error', r'HTTP error|status code [45]\d\d'),
                ('Connection Error', r'ConnectionError|Connection refused|Could not resolve host'),
                ('SSL Error', r'SSL|certificate|self signed certificate'),
                ('Redirect Error', r'TooManyRedirects|redirect|unsupported'),
                ('Content Error', r'Failed to extract|No meaningful content|Content too large'),
                ('Parsing Error', r'Error finding content|HTML processing failed|extraction error')
            ]
            
            error_counts = {}
            for error_type, pattern in error_patterns:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                error_counts[error_type] = len(matches)
                
            # Mostrar resultados
            print("\nAnálisis de web_scraper_error.log:")
            print("="*40)
            for error_type, count in error_counts.items():
                print(f"{error_type}: {count} ocurrencias")
                
            # Buscar los errores más recientes
            log_lines = log_content.splitlines()
            recent_errors = []
            for i in range(len(log_lines) - 1, max(0, len(log_lines) - 20), -1):
                line = log_lines[i]
                if 'ERROR' in line:
                    recent_errors.append(line)
                if len(recent_errors) >= 5:
                    break
                    
            print("\nÚltimos 5 errores:")
            print("="*40)
            for error in reversed(recent_errors):
                print(f"- {error}")
                
            return True
        else:
            print("No se encontró el archivo web_scraper_error.log")
            return False
    except Exception as e:
        print(f"Error al verificar web_scraper_error.log: {str(e)}")
        return False

def run_tests():
    """
    Ejecutar pruebas con todos los textos y analizar resultados
    """
    results = []
    
    print("="*50)
    print("PRUEBA DE ENDPOINT /check")
    print("="*50)
    
    # Ejecutar pruebas con cada texto
    for test in TEST_TEXTS:
        print(f"\nProbando con: {test['name']}")
        print("-"*30)
        
        success, message = check_endpoint(test['text'])
        
        result = {
            "name": test['name'],
            "success": success,
            "message": message
        }
        results.append(result)
        
        print(f"Resultado: {'ÉXITO' if success else 'FALLO'}")
        print(f"Mensaje: {message}")
    
    # Mostrar resumen
    print("\n"+"="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)
    
    successful_tests = sum(1 for r in results if r['success'])
    print(f"Pruebas exitosas: {successful_tests}/{len(results)}")
    
    print("\nResultados por prueba:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['name']}: {result['message']}")
    
    # Verificar log de web_scraper
    check_web_scraper_log()
    
    print("\nLogs completos disponibles en endpoint_debug.log")
    
if __name__ == "__main__":
    run_tests()