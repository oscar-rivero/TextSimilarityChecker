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
        results = check_plagiarism(text)
        
        # Store results in session for report generation
        request.session['check_results'] = results
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
