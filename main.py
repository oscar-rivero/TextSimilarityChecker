import os
import logging
import nltk
import sys
from gunicorn.app.wsgiapp import WSGIApplication

# Download required NLTK datasets
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    download_nltk_data()
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

# Django WSGI application
# This allows us to run the Django app with Gunicorn
from plagiarism_detector_project.wsgi import application as app

if __name__ == "__main__":
    # Run the app
    sys.argv = ["gunicorn", "--bind", "0.0.0.0:5000", "--reload", "main:app"]
    WSGIApplication().run()
