import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def initialize():
    nltk.download('vader_lexicon')

    
def make_sentiment_df(df):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    for article in df:
        try:
            title = df["Title"]
            upvote = df["Upvote Ratio"]
            text = df["Text"]
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
                text_compound = 0
                text_pos = 0
                text_neu = 0
                text_neg = 0
            
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
    df.describe()
    
def sentiment_scaler(score):    
    result = 0
    if score >= 0.05:
        result = 1
    elif score < 0.05:
        result = -1
        
    return result

