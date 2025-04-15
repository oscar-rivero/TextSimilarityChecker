"""
Este script modifica temporalmente la configuración de Django para desactivar
la protección CSRF durante pruebas locales.
"""

import os
import re

# Archivos a modificar
SETTINGS_FILE = 'plagiarism_detector_project/settings.py'

def disable_csrf():
    """Desactiva la protección CSRF en la configuración de Django"""
    print("Desactivando protección CSRF para pruebas locales...")
    
    if not os.path.exists(SETTINGS_FILE):
        print(f"Error: No se encontró el archivo {SETTINGS_FILE}")
        return False
    
    # Leer el archivo de configuración
    with open(SETTINGS_FILE, 'r') as f:
        content = f.read()
    
    # Buscar la sección de middleware
    middleware_match = re.search(r'MIDDLEWARE\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not middleware_match:
        print("Error: No se encontró la configuración MIDDLEWARE")
        return False
    
    middleware_content = middleware_match.group(1)
    
    # Verificar si el middleware CSRF está presente
    if 'django.middleware.csrf.CsrfViewMiddleware' in middleware_content:
        # Comentar el middleware CSRF
        modified_middleware = middleware_content.replace(
            "'django.middleware.csrf.CsrfViewMiddleware'", 
            "# 'django.middleware.csrf.CsrfViewMiddleware'  # Comentado temporalmente para pruebas"
        )
        
        # Reemplazar en el contenido original
        modified_content = content.replace(middleware_content, modified_middleware)
        
        # Guardar los cambios
        with open(SETTINGS_FILE, 'w') as f:
            f.write(modified_content)
        
        print("Protección CSRF desactivada correctamente.")
        return True
    else:
        print("Nota: El middleware CSRF ya está desactivado o no está presente.")
        return True

def enable_csrf():
    """Restaura la protección CSRF en la configuración de Django"""
    print("Restaurando protección CSRF...")
    
    if not os.path.exists(SETTINGS_FILE):
        print(f"Error: No se encontró el archivo {SETTINGS_FILE}")
        return False
    
    # Leer el archivo de configuración
    with open(SETTINGS_FILE, 'r') as f:
        content = f.read()
    
    # Buscar la sección de middleware
    middleware_match = re.search(r'MIDDLEWARE\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not middleware_match:
        print("Error: No se encontró la configuración MIDDLEWARE")
        return False
    
    middleware_content = middleware_match.group(1)
    
    # Verificar si el middleware CSRF está comentado
    if "# 'django.middleware.csrf.CsrfViewMiddleware'" in middleware_content:
        # Descomentar el middleware CSRF
        modified_middleware = middleware_content.replace(
            "# 'django.middleware.csrf.CsrfViewMiddleware'  # Comentado temporalmente para pruebas", 
            "'django.middleware.csrf.CsrfViewMiddleware'"
        )
        
        # Reemplazar en el contenido original
        modified_content = content.replace(middleware_content, modified_middleware)
        
        # Guardar los cambios
        with open(SETTINGS_FILE, 'w') as f:
            f.write(modified_content)
        
        print("Protección CSRF restaurada correctamente.")
        return True
    else:
        print("Nota: El middleware CSRF no estaba comentado o no se encontró.")
        return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python disable_csrf.py [disable|enable]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == "disable":
        disable_csrf()
    elif action == "enable":
        enable_csrf()
    else:
        print(f"Acción desconocida: {action}")
        print("Uso: python disable_csrf.py [disable|enable]")
        sys.exit(1)