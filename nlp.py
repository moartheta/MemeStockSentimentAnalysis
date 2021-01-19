import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
def initialize():
    nltk.download('vader_lexicon')

    
def make_sentiment_df(df, src = "Reddit"):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    #adjust later for other API's
    if src.lower() == "reddit":
        for index,row in df.iterrows():
            try:
                title = row["Title"]
                upvote = row["Upvote Ratio"]
                text = row["Text"]
                sentiment = analyzer.polarity_scores(title)
                title_compound = sentiment["compound"]
                title_pos = sentiment["pos"]
                title_neu = sentiment["neu"]
                title_neg = sentiment["neg"]
                sentiment = analyzer.polarity_scores(text)
                #To account for blank text values
                if len(text) > 0:
                    text_compound = sentiment["compound"]
                    text_pos = sentiment["pos"]
                    text_neu = sentiment["neu"]
                    text_neg = sentiment["neg"]
                else:
                    text_compound = np.nan
                    text_pos = np.nan
                    text_neu = np.nan
                    text_neg = np.nan

                sentiments.append({
                    "Title": title,
                    "Upvote_Ratio": upvote,
                    "Title_Compound": title_compound,
                    "Title_Positive": title_pos,
                    "Title_Negative": title_neg,
                    "Title_Neutral" : title_neu,
                    "Text_Compound" : text_compound,
                    "Text_Positive" : text_pos,
                    "Text_Negative" : text_neg,
                    "Text_Neutral" : text_neu
                })
            except AttributeError:
                pass
        
    df = pd.DataFrame(sentiments)
    #reorder columns if needed
    #cols = ["date", "text", ...]
    #df = df[cols]
    return df 

def show_stats(df):
    return df.describe()
    
def sentiment_scaler(score):    
    result = 0
    if score >= 0.05:
        result = 1
    elif score < 0.05:
        result = -1
        
    return result

