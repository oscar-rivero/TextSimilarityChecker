import os
import logging
from flask import Flask, render_template, request, jsonify, session
from plagiarism_detector import check_plagiarism, generate_report
import json

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development_secret_key")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the main page with the input form."""
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    """Process the submitted text and check for plagiarism."""
    try:
        # Get text from form
        text = request.form.get('text', '')
        
        if not text.strip():
            return render_template('index.html', error="Please enter some text to check for plagiarism.")
        
        # Perform plagiarism check
        logger.debug(f"Checking plagiarism for text of length: {len(text)}")
        results = check_plagiarism(text)
        
        # Store results in session for report generation
        session['check_results'] = results
        session['original_text'] = text
        
        # Generate report data
        report_data = generate_report(text, results)
        
        return render_template('results.html', 
                              results=results, 
                              original_text=text, 
                              report_data=report_data)
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error during plagiarism check: {error_message}")
        
        # Provide a more user-friendly message for rate limit errors
        if "429" in error_message or "rate limit" in error_message.lower():
            error_msg = "The search service is currently experiencing high traffic. Please try again in a few minutes or with different text."
        else:
            error_msg = f"An error occurred during the plagiarism check. Please try again."
            
        return render_template('index.html', error=error_msg)

@app.route('/report', methods=['GET'])
def report():
    """Generate a downloadable report."""
    try:
        # Get results from session
        results = session.get('check_results', None)
        original_text = session.get('original_text', '')
        
        if not results:
            return jsonify({"error": "No plagiarism check results found. Please perform a check first."}), 400
        
        # Generate report data
        report_data = generate_report(original_text, results)
        
        return jsonify(report_data)
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
