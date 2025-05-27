import praw
import pandas as pd
from datetime import datetime
from psaw import PushshiftAPI


reddit = praw.Reddit(
    client_id="4iDE06oRtkTPDv46eVRjyA",
    client_secret="ggeSKBmo5SNJrxIUySTr1lw_9qO09Q",
    user_agent="AmithETFExtractor"
)
api = PushshiftAPI(reddit)

# Test 1: Print your Reddit username (will work if you're authenticated)
try:
    print("Logged in as:", reddit.user.me())
except Exception as e:
    print("Error during login:", e)

# Your list of subreddits
subreddits = ["IndianStockMarket", "mutualfunds", "personalfinance", "investing"]

# Date range
start_date = int(datetime(2024, 9, 1).timestamp())
end_date = int(datetime(2025, 4, 30).timestamp())

# Output data
all_data = []

for sub in subreddits:
    print(f"Fetching posts from r/{sub}...")
    submissions = api.search_submissions(after=start_date,
                                         before=end_date,
                                         subreddit=sub,
                                         filter=['title', 'score', 'author', 'created_utc', 'subreddit'],
                                         limit=100)

    for post in submissions:
        all_data.append({
            "Subreddit": post.subreddit,
            "Title": post.title,
            "Author": str(post.author),
            "Score": post.score,
            "Created Date": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
        })

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv("reddit_posts_etf_sept2024_apr2025.csv", index=False)
print("Data saved to reddit_posts_etf_sept2024_apr2025.csv")
