"""
Script para diagnosticar errores internos del servidor en la aplicación Django.
Este script analizará los logs del servidor y ayudará a identificar la causa raíz.
"""

import os
import re
import sys
import traceback
import importlib
from pprint import pprint
import logging

# Configurar logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_django_config():
    """Verificar la configuración de Django"""
    try:
        # Verificar que Django está instalado
        import django
        logger.info(f"Django versión: {django.get_version()}")
        
        # Verificar el directorio actual
        logger.info(f"Directorio actual: {os.getcwd()}")
        
        # Verificar que settings.py existe
        settings_path = 'plagiarism_detector_project/settings.py'
        if os.path.exists(settings_path):
            logger.info(f"Archivo de configuración encontrado: {settings_path}")
            
            # Leer y analizar settings.py
            with open(settings_path, 'r') as f:
                settings_content = f.read()
            
            # Verificar URLs configuradas
            urls_path = 'plagiarism_detector_project/urls.py'
            logger.info(f"Analizando configuración de URLs en {urls_path}")
            
            if os.path.exists(urls_path):
                with open(urls_path, 'r') as f:
                    urls_content = f.read()
                
                # Buscar patrones comunes de URL
                url_patterns = re.findall(r'path\([\'"]([^\'"]+)[\'"]', urls_content)
                logger.info(f"Patrones de URL encontrados: {url_patterns}")
                
                # Verificar si el endpoint /check/ está correctamente configurado
                check_url_configured = any(pattern == 'check/' for pattern in url_patterns)
                logger.info(f"Endpoint 'check/' configurado: {check_url_configured}")
            else:
                logger.error(f"Archivo de URLs no encontrado: {urls_path}")
        else:
            logger.error(f"Archivo de configuración no encontrado: {settings_path}")
        
        # Verificar middleware
        middleware_match = re.search(r'MIDDLEWARE\s*=\s*\[(.*?)\]', settings_content, re.DOTALL)
        if middleware_match:
            middleware_str = middleware_match.group(1)
            middleware_items = re.findall(r'[\'"]([^\'"]+)[\'"]', middleware_str)
            logger.info("Middleware configurado:")
            for item in middleware_items:
                logger.info(f"  - {item}")
            
            # Verificar CSRF middleware
            csrf_middleware = 'django.middleware.csrf.CsrfViewMiddleware'
            csrf_enabled = csrf_middleware in middleware_items
            logger.info(f"CSRF middleware habilitado: {csrf_enabled}")
        else:
            logger.error("No se pudo encontrar la configuración de MIDDLEWARE")
    
    except ImportError as e:
        logger.error(f"Error importando Django: {str(e)}")
    except Exception as e:
        logger.error(f"Error verificando configuración de Django: {str(e)}")
        logger.error(traceback.format_exc())

def check_app_routes():
    """Verificar las rutas de la aplicación"""
    try:
        # Verificar archivos de la aplicación
        app_path = 'plagiarism_app'
        views_path = f'{app_path}/views.py'
        
        if os.path.exists(views_path):
            logger.info(f"Archivo de vistas encontrado: {views_path}")
            
            with open(views_path, 'r') as f:
                views_content = f.read()
            
            # Buscar funciones de vista definidas
            view_functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', views_content)
            logger.info(f"Funciones de vista encontradas: {view_functions}")
            
            # Verificar si la función 'check' está definida
            check_view_defined = 'check' in view_functions
            logger.info(f"Función de vista 'check' definida: {check_view_defined}")
            
            # Buscar código de la vista 'check'
            check_view_match = re.search(r'def\s+check\s*\(.*?:.*?\n(.*?)(?:def\s+|$)', views_content, re.DOTALL)
            if check_view_match:
                check_view_code = check_view_match.group(1).strip()
                logger.info(f"Código de la vista 'check' (primeras 500 caracteres):")
                logger.info(check_view_code[:500] + ("..." if len(check_view_code) > 500 else ""))
            else:
                logger.error("No se pudo encontrar el código de la vista 'check'")
        else:
            logger.error(f"Archivo de vistas no encontrado: {views_path}")
        
        # Verificar URLs de la aplicación
        urls_path = f'{app_path}/urls.py'
        if os.path.exists(urls_path):
            logger.info(f"Archivo de URLs de la aplicación encontrado: {urls_path}")
            
            with open(urls_path, 'r') as f:
                app_urls_content = f.read()
            
            # Buscar patrones de URL específicos de la aplicación
            app_url_patterns = re.findall(r'path\([\'"]([^\'"]+)[\'"]', app_urls_content)
            logger.info(f"Patrones de URL de la aplicación: {app_url_patterns}")
            
            # Verificar configuración del endpoint 'check'
            check_url_in_app = any(pattern == 'check/' for pattern in app_url_patterns)
            logger.info(f"Endpoint 'check/' configurado en la aplicación: {check_url_in_app}")
        else:
            logger.error(f"Archivo de URLs de la aplicación no encontrado: {urls_path}")
    
    except Exception as e:
        logger.error(f"Error verificando rutas de la aplicación: {str(e)}")
        logger.error(traceback.format_exc())

def test_import_modules():
    """Probar importación de módulos críticos"""
    critical_modules = [
        'web_scraper', 
        'text_classifier', 
        'plagiarism_detector', 
        'semantic_comparison'
    ]
    
    for module_name in critical_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"Módulo '{module_name}' importado correctamente")
            
            # Listar funciones públicas del módulo
            public_functions = [name for name in dir(module) 
                             if callable(getattr(module, name)) and not name.startswith('_')]
            logger.info(f"Funciones públicas en '{module_name}': {public_functions[:5]}{'...' if len(public_functions) > 5 else ''}")
            
        except ImportError as e:
            logger.error(f"Error importando módulo '{module_name}': {str(e)}")
        except Exception as e:
            logger.error(f"Error al inspeccionar módulo '{module_name}': {str(e)}")

def check_view_function():
    """Verificar la función 'check' directamente"""
    try:
        # Configurar entorno Django para pruebas
        import django
        import os
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plagiarism_detector_project.settings')
        django.setup()
        
        # Importar función de vista
        from plagiarism_app.views import check
        
        logger.info("Función de vista 'check' importada correctamente")
        logger.info(f"Firma de la función: {check.__code__.co_varnames[:check.__code__.co_argcount]}")
        logger.info(f"Número de línea: {check.__code__.co_firstlineno}")
        
    except ImportError as e:
        logger.error(f"Error importando función 'check': {str(e)}")
    except Exception as e:
        logger.error(f"Error verificando función 'check': {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Función principal"""
    print("=" * 80)
    print("DIAGNÓSTICO DE ERRORES INTERNOS DEL SERVIDOR")
    print("=" * 80)
    
    print("\n1. Verificando configuración de Django...")
    check_django_config()
    
    print("\n2. Verificando rutas de la aplicación...")
    check_app_routes()
    
    print("\n3. Probando importación de módulos críticos...")
    test_import_modules()
    
    print("\n4. Verificando función de vista 'check'...")
    check_view_function()
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    main()