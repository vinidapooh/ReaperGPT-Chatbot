import requests
from bs4 import BeautifulSoup
import time
import json
import re
import os  # FIX 1: Added missing import

# --- CONFIGURATION ---
BASE_URL = "https://forum.cockos.com/forumdisplay.php?f=19"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
BATCH_SIZE = 20  # How many pages to learn daily
WAIT_TIME = 5    # Seconds between requests

def get_next_page_range(batch_size=5):
    checkpoint_file = "scrape_checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            last_page = json.load(f).get('last_page', 0)
    else:
        last_page = 0
        
    start_page = last_page + 1
    end_page = last_page + batch_size
    
    # Update checkpoint for next time
    with open(checkpoint_file, 'w') as f:
        json.dump({'last_page': end_page}, f)
        
    return start_page, end_page

def get_thread_links(page_url):
    """Collects thread URLs from a specific forum page."""
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=10)
        if response.status_code != 200: return []
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', id=lambda x: x and x.startswith('thread_title_'))
        return [{"title": l.get_text(strip=True), "url": "https://forum.cockos.com/" + l.get('href').split('&s=')[0]} for l in links]
    except Exception as e:
        print(f"Error fetching links: {e}")
        return []

def scrape_thread_content(url):
    """Extracts and cleans posts from a specific thread."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = soup.find_all('div', id=lambda x: x and x.startswith('post_message_'))
        cleaned_posts = []
        for post in posts:
            text = post.get_text(separator=' ', strip=True)
            text = re.sub(r'Quote:.*?Originally Posted by.*?\s', '', text)
            if len(text) > 80: cleaned_posts.append(text)
        return "\n\n---NEXT POST---\n\n".join(cleaned_posts)
    except Exception: return None

def main():
    # FIX 2: Load existing data so we APPEND, not overwrite
    if os.path.exists('forum_data.json'):
        with open('forum_data.json', 'r', encoding='utf-8') as f:
            all_extracted_data = json.load(f)
    else:
        all_extracted_data = []

    # FIX 3: Actually use the checkpoint logic
    start_page, end_page = get_next_page_range(BATCH_SIZE)
    print(f"🚀 Starting Reaper Forum ETL (Pages {start_page} to {end_page})...")
    
    for page in range(start_page, end_page + 1):
        print(f"\n📄 Processing Page {page}...")
        page_url = f"{BASE_URL}&page={page}"
        threads = get_thread_links(page_url)
        
        for thread in threads:
            # Check if we already have this URL to avoid duplicates
            if any(d['url'] == thread['url'] for d in all_extracted_data):
                continue
                
            content = scrape_thread_content(thread['url'])
            if content:
                all_extracted_data.append({
                    "title": thread['title'],
                    "url": thread['url'],
                    "content": content,
                    "source": "Reaper Forum",
                    "metadata": {"forum_page": page, "ingestion_date": time.strftime("%Y-%m-%d")}
                })
            time.sleep(WAIT_TIME)

    if all_extracted_data:
        with open('forum_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_extracted_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ SUCCESS: Total threads in database: {len(all_extracted_data)}")
    else:
        print("\n❌ No new data found.")

if __name__ == "__main__":
    main()