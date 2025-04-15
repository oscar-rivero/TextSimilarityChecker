import os
import logging
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
from django.contrib import messages

# Import plagiarism detector functionality
from plagiarism_detector import check_plagiarism, generate_report

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def index(request):
    """Render the main page with the input form."""
    return render(request, 'index.html')

@csrf_protect
def check(request):
    """Process the submitted text and check for plagiarism."""
    if request.method != 'POST':
        return redirect('index')
    
    try:
        # Get text from form
        text = request.POST.get('text', '')
        
        if not text.strip():
            messages.error(request, "Please enter some text to check for plagiarism.")
            return redirect('index')
        
        # Perform plagiarism check
        logger.debug(f"Checking plagiarism for text of length: {len(text)}")
        
        # Verificar si la sesión está inicializada correctamente
        if not request.session.session_key:
            request.session.create()
            logger.debug("Creando nueva sesión para el usuario")
            
        results = check_plagiarism(text)
        
        # Convertir resultados a JSON serializable para evitar errores de sesión
        # Esto soluciona el problema con AnonymousUser
        serializable_results = []
        for result in results:
            # Crear copia serializable del resultado
            serializable_result = {
                "source": {
                    "title": result["source"]["title"],
                    "url": result["source"]["url"],
                    "snippet": result["source"]["snippet"]
                },
                "similarity": result["similarity"],
                "matches": result["matches"],
                "category_tag": result.get("category_tag", "General"),
                "relevance_score": result.get("relevance_score", 0),
                "best_paragraph": {
                    "content": result["best_paragraph"]["content"],
                    "similarity": result["best_paragraph"]["similarity"]
                }
            }
            serializable_results.append(serializable_result)
        
        # Store results in session for report generation
        request.session['check_results'] = serializable_results
        request.session['original_text'] = text
        
        # Generate report data
        report = generate_report(text, results)
        
        return render(request, 'results.html', {
            'results': results, 
            'original_text': text, 
            'report': report
        })
    
    except Exception as e:
        logger.error(f"Error during plagiarism check: {str(e)}")
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('index')

def report(request):
    """Generate a downloadable report."""
    try:
        # Get results from session
        results = request.session.get('check_results', None)
        original_text = request.session.get('original_text', '')
        
        if not results:
            return JsonResponse({"error": "No plagiarism check results found. Please perform a check first."}, status=400)
        
        # Generate report data
        report = generate_report(original_text, results)
        
        return JsonResponse(report)
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
