import requests
import time
import json
import os

def fetch_etf_questions(site,tagged,pagesize,max_pages,output_dir,output_filename):
    base_url = "https://api.stackexchange.com/2.3/questions"
    questions = []
    page = 1

    print(f"Fetching ETF-related questions from {site}.stackexchange.com...")

    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_path = os.path.join(output_dir, output_filename)

    while page <= max_pages:
        params = {
            "order": "desc",
            "sort": "activity",
            "tagged": tagged,
            "site": site,
            "filter": "withbody",
            "pagesize": pagesize,
            "page": page
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code}")
            break

        data = response.json()
        questions.extend(data.get("items", []))
        print(f"Fetched page {page} with {len(data.get('items', []))} questions.")

        if not data.get("has_more"):
            break

        page += 1
        time.sleep(1.2)  # Respect API rate limit

    # Save to JSON file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(questions)} questions to {output_path}")
    return questions

# Run the script
if __name__ == "__main__":
    output_dir='Data/Raw/'
    output_filename='stackexchange_prompt.json'
    site='money'
    tagged='etf'
    pagesize=100
    max_pages=5
    fetch_etf_questions(site,tagged,pagesize,max_pages,output_dir,output_filename)
