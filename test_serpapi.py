import os
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_serpapi():
    """
    Test the SerpAPI functionality and log the response structure.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        logger.error("No SERPAPI_KEY found in environment variables.")
        return False
        
    # Test query
    query = "python plagiarism detection algorithm"
    
    # Test parameters
    params = {
        "q": query,
        "api_key": api_key,
        "num": 10,
        "gl": "us",  # Set to US for more reliable results
        "hl": "en"   # Set to English
    }
    
    try:
        logger.info(f"Sending search request for query: {query}")
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        
        # Load the response as JSON
        response_data = response.json()
        
        # Check for errors
        if "error" in response_data:
            logger.error(f"SerpAPI returned error: {response_data['error']}")
            return False
            
        # Log the top-level keys in the response
        logger.info(f"Response keys: {list(response_data.keys())}")
        
        # Look for search results under different possible keys
        result_keys = ["organic_results", "results", "search_results"]
        found_results = False
        
        for key in result_keys:
            if key in response_data and response_data[key]:
                results = response_data[key]
                logger.info(f"Found {len(results)} results under key '{key}'")
                
                # Log the first result structure
                if results:
                    logger.info(f"First result structure: {list(results[0].keys())}")
                    found_results = True
                    break
        
        if not found_results:
            logger.warning("No search results found in any expected keys")
            
        # Save the entire response to a file for inspection
        with open("serpapi_response.json", "w") as f:
            json.dump(response_data, f, indent=2)
            
        logger.info("Response saved to serpapi_response.json")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Error making request to SerpAPI: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_serpapi()
    if success:
        logger.info("SerpAPI test completed successfully")
    else:
        logger.error("SerpAPI test failed")