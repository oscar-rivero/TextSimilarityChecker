"""
Script para probar directamente la funcionalidad de web_scraper.py
"""

import sys
from web_scraper import get_website_text_content
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_url(url):
    """Prueba la extracción de contenido para una URL específica"""
    print(f"\nProbando extracción para: {url}")
    print("-" * 60)
    
    try:
        # Intentar extraer contenido
        content = get_website_text_content(url)
        
        # Verificar si se obtuvo contenido
        if content:
            content_length = len(content)
            print(f"✓ Éxito! Se obtuvieron {content_length} caracteres")
            
            # Mostrar fragmento del contenido
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"\nMuestra de contenido:")
            print("-" * 30)
            print(preview)
            print("-" * 30)
            
            return True
        else:
            print("✗ No se pudo extraer contenido (cadena vacía)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # URLs de prueba
    test_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://www.bbc.com/news",
        "https://www.nytimes.com",
        "https://www.sciencedirect.com/science/article/pii/S0048969720359209"  # URL problemática
    ]
    
    # Si se proporciona una URL específica, usar esa
    if len(sys.argv) > 1:
        test_urls = [sys.argv[1]]
    
    print("=" * 60)
    print("PRUEBA DE EXTRACCIÓN DE CONTENIDO WEB")
    print("=" * 60)
    
    # Probar cada URL
    results = []
    for url in test_urls:
        success = test_url(url)
        results.append((url, success))
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total de pruebas: {len(results)}")
    print(f"Exitosas: {sum(1 for _, success in results if success)}")
    print(f"Fallidas: {sum(1 for _, success in results if not success)}")
    
    print("\nResultados por URL:")
    for url, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {url}")