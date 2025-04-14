import requests
from bs4 import BeautifulSoup
import json

def extract_categories():
    # Get Wikipedia main categories page
    url = "https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories"
    response = requests.get(url)
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all categories listed on the page
    categories = {}
    
    # Find main content div
    content_div = soup.find("div", {"id": "mw-content-text"})
    
    if content_div:
        # Get all headings in the content
        headings = content_div.find_all(['h2', 'h3'])
        
        current_main_cat = None
        
        for heading in headings:
            # Check if it's a main heading (h2)
            if heading.name == 'h2':
                heading_text = heading.get_text().strip()
                if 'References' not in heading_text and 'Contents' not in heading_text:
                    current_main_cat = heading_text
                    categories[current_main_cat] = []
            
            # Check if it's a subheading (h3) and we have a current main category
            elif heading.name == 'h3' and current_main_cat:
                heading_text = heading.get_text().strip()
                
                # Find the next unordered list with categories
                next_list = heading.find_next('ul')
                if next_list:
                    subcategories = []
                    list_items = next_list.find_all('li')
                    
                    for item in list_items:
                        links = item.find_all('a')
                        for link in links:
                            if 'Category:' in link.get('href', ''):
                                cat_name = link.get_text().strip()
                                subcategories.append(cat_name)
                    
                    if subcategories:
                        categories[current_main_cat].append({
                            'name': heading_text,
                            'subcategories': subcategories
                        })
    
    # Direct category extraction from links
    all_categories = []
    cat_links = soup.find_all('a', href=lambda href: href and '/wiki/Category:' in href)
    
    for link in cat_links:
        cat_name = link.get_text().strip()
        if cat_name and cat_name not in all_categories:
            all_categories.append(cat_name)
    
    return {
        'structured_categories': categories,
        'all_categories': all_categories
    }

if __name__ == "__main__":
    categories = extract_categories()
    with open('wiki_categories.json', 'w') as f:
        json.dump(categories, f, indent=2)
    
    print(f"Found {len(categories['all_categories'])} categories")
    print("Sample categories:")
    for cat in categories['all_categories'][:20]:
        print(f"- {cat}")