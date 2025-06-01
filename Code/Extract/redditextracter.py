import praw
import json
import os
from datetime import datetime, timezone

def fetch_recent_reddit_posts(output_dir,limit_per_sub):
    reddit = praw.Reddit(
        client_id="4iDE06oRtkTPDv46eVRjyA",
        client_secret="ggeSKBmo5SNJrxIUySTr1lw_9qO09Q",
        user_agent="AmithETFExtractor"
    )

    subreddits = ["IndianStockMarket", "mutualfunds", "personalfinance", "investing"]
    all_data = []

    for sub in subreddits:
        print(f"Fetching {limit_per_sub} posts from r/{sub}...")
        for submission in reddit.subreddit(sub).new(limit=limit_per_sub):
            if "etf" in submission.title.lower():  # Case-insensitive filter
                created_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime('%Y-%m-%d')
                all_data.append({
                "subreddit": submission.subreddit.display_name,
                "title": submission.title,
                "author": str(submission.author),
                "score": submission.score,
                "created_date": created_date,
                 })

    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f"reddit_posts_etf_{now}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_data)} posts to {output_path}")

if __name__ == "__main__":
    output_dir='Data/Raw/'
    limit_per_sub=100
    fetch_recent_reddit_posts(output_dir,limit_per_sub)
