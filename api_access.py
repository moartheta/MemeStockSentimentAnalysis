import os
import praw
import quandl
import pandas as pd
from dotenv import load_dotenv
import quandl
import alpaca_trade_api as tradeapi

load_dotenv()

def OAuth(api):

    if api == "reddit":
        
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
    
    elif api == "quandl":
        
        quandl_key = os.getenv("quandl_key")
    
        if type(quandl_key) == str:
            print("Quandl Key Loaded!")
        else:
            print("Quandl Key Error!")
            
        quandl.ApiConfig.api_key = quandl_key
        
        return quandl
    
    elif api == "alpaca":
        
        alpaca_api_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    
        if type(alpaca_api_key) == str:
            print("Alpaca Key Loaded!")
        else:
            print("Alpaca Key Error!")
        if type(alpaca_secret_key) == str:
            print("Alpaca Secret Key Loaded!")
        else:
            print("Alpaca Secret Key Error!")
        if type(alpaca_api_key) == str and type(alpaca_secret_key) == str:
            print("Alpaca API Successfully Initialized!")
        else:
            print("API Initialization Error!")
            
        alpaca = tradeapi.REST( alpaca_api_key,
                                alpaca_secret_key,
                                api_version="v2")
        
        return alpaca

    elif api == "twitter":
        consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
        consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET_KEY")
        access_key = os.getenv("TWITTER_ACCESS_TOKEN")
        access_secret = os.getenv("TWITTER_ACCESS_SECRET_TOKEN")
    
        if type(consumer_key) == str:
            print("Twitter Conusmer Key Loaded!")
        else:
            print("Twitter Consumer Key Error!")
        if type(consumer_secret) == str:
            print("Twitter Consumer Secret Key Loaded!")
        else:
            print("Twitter Consumer Secret Key Error!")
        if type(access_key) == str:
            print("Twitter Access Key Loaded!")
        else:
            print("Twitter Access Key Error!")
        if type(access_secret) == str:
            print("Twitter Access Secret Key Loaded!")
        else:
            print("Twitter Access Secret Key Error!")
        
        return twitter
    
    else:
        print("Incorrect API Name")


def fetch_subreddit(reddit_access, subreddit):
    
    subreddit = reddit_access.subreddit(subreddit)
    print(f'You have chosen r/{subreddit.title} as your subreddit.  May god have mercy on your soul.')
    
    return subreddit


def fetch_reddit_comments(api_connection, search, subreddit, max_response):
    
    posts = api_connection.search_submissions(q=search, subreddit=subreddit)

    max_response_cache = max_response
    
    cache = []

    for c in posts:
        cache.append(c)

        if len(cache) >= max_response_cache:
            break
    
    combined = []

    for submission in cache:
        data = []
        data.append(submission.title)
        data.append(submission.upvote_ratio)
        data.append(submission.selftext)
        data.append(submission.created_utc)
        combined.append(data)
    
    reddit_posts = pd.DataFrame(combined, columns=['Title', 'Upvote Ratio', 'Text', 'Date'])
    reddit_posts['Date'] = pd.to_datetime(reddit_posts['Date'], unit = 's')
    
    cleaned_comments = reddit_posts[(reddit_posts['Text'] != '[removed]')]
    cleaned_comments = cleaned_comments[(cleaned_comments['Text'] != '[deleted]')]
    
    length = len(reddit_posts.index)
    
    print(f"Your request returned {length} results.")
    
    return cleaned_comments
        
def fetch_alpaca(alpaca, tickers, start_date, end_date, period):
    
    start = pd.Timestamp(start_date, tz="America/New_York").isoformat()
    end = pd.Timestamp(end_date, tz="America/New_York").isoformat()
    timeframe = period
    
    stock_data = alpaca.get_barset(
    tickers,
    period,
    start = start,
    end = end).df
    
    stock_data = stock_data.iloc[:, stock_data.columns.get_level_values(1)=='close']
    
    return stock_data
    
def fetch_quandl(quandl, tickers, start_date, end_date):
    
    data = quandl.get(tickers, start_date=start_date, end_date=end_date)
    
    ticker_list = []
    
    for ticker in tickers:
        new_ticker = ticker + " - Settle"
        ticker_list.append(new_ticker)
    
    data = data[ticker_list]
    
    return data