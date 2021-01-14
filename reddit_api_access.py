import os
import praw
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def reddit_OAuth():

    reddit_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_secret_key = os.getenv("REDDIT_SECRET_ID")
    reddit_username = os.getenv("REDDIT_USER_AGENT")

    if type(reddit_id) == str:
        print("Reddit Key Loaded!")
    else:
        print("Reddit Key Error!")
    if type(reddit_secret_key) == str:
        print("Reddit Secret Key Loaded!")
    else:
        print("Reddit Secret Key Error!")
    if type(reddit_username) == str:
        print("Reddit Username Loaded!")
    else:
        print("Reddit Username Error!")
    
    print('Authorization Successful.  Connected to Reddit API.')
    
    reddit_auth = praw.Reddit(client_id =reddit_id, 
                              client_secret =reddit_secret_key, 
                              user_agent =reddit_username)

    return reddit_auth

def fetch_subreddit(reddit_access, subreddit):
    
    subreddit = reddit_access.subreddit(subreddit)
    print(f'You have chosen r/{subreddit.title} as your subreddit.  May god have mercy on your soul.')
    
    return subreddit


def fetch_reddit_comments(subreddit, comment_limit):
    
    combined = []
    
    for submission in subreddit.top(limit=comment_limit):
        data = []
        data.append(submission.title)
        data.append(submission.upvote_ratio)
        data.append(submission.selftext)
        data.append(submission.created_utc)
        combined.append(data)
    
    reddit_posts = pd.DataFrame(combined, columns=['Title', 'Upvote Ratio', 'Text', 'Date'])
    reddit_posts['Date'] = pd.to_datetime(reddit_posts['Date'], unit = 's')
    
    return reddit_posts