import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from pathlib import Path


def combine(file1, file2):
    f1 = Path(file1)
    f2 = Path(file2)
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    
    df = pd.concat([df1,df2], axis = 0)
    
    df = df.groupby(["Created"]).mean()
    df.sort_index(inplace = True)
    
    return df

def sentiment_cleaner(df):
    """
    Fixes the formatting issues from the 'Sentiment' field from the API
    Returns the original dataframe with the cleaned data
    """
    df = df.copy()

    df['Sentiment'] = df['Sentiment'].str.replace('{','')
    df['Sentiment'] = df['Sentiment'].str.replace('}','')
    df['Sentiment'] = df['Sentiment'].str.replace("'",'')
    df['Sentiment'] = df['Sentiment'].str.replace("basic",'')
    df['Sentiment'] = df['Sentiment'].str.replace(": ",'')
        
    return df


def fix_date(df):
    """
    Fixes the unix date and returns the original dataframe with the day in YYYY-MM-DD format
    """
        
    df = df.copy()
        
    df['Created'] = pd.to_datetime(df['Created'], utc = True)
    df['Created'] = df['Created'].dt.date
        
    return df

class NLT:
    def __init__(self):
        pass
    
    def update_vader(self):
        nltk.download('vader_lexicon')
        

    def make_sentiment_df(self,df, src = "Reddit"):
        """
        This is where the money is made.  Takes dataframe and optional source (reddit or twits)
        adds sentiment analysis fields to the dataframe and returns a new dataframe
        """
        
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        #df = sentiment_cleaner(df)
        #adjust later for other API's
        if src.lower() == "reddit":
            #for index,row in df.iterrows():
            #This one is faster if it works
            for index, row in df.iterrows():
                try:
                    title = row["Title"]
                    created = row["Date"]
                    upvote = row["Upvote Ratio"]
                    sentiment = analyzer.polarity_scores(title)
                    title_compound = sentiment["compound"]
                    title_pos = sentiment["pos"]
                    title_neu = sentiment["neu"]
                    title_neg = sentiment["neg"]

                    sentiments.append({
                        "Text": title,
                        "Created" : created,
                        "Upvote_Ratio": upvote,
                        "NLTK_Compound": title_compound,
                        "NLTK_Positive": title_pos,
                        "NLTK_Negative": title_neg,
                        "NLTK_Neutral" : title_neu,
                    })
                except AttributeError:
                    pass
        elif src.lower() == "twits":
            df = sentiment_cleaner(df)
            df = fix_date(df)
            for index, row in df.iterrows():
                try:
                    text = row["Text"]
                    created = row["Created"]
                    likes = row["Likes"]
                    sent = row["Sentiment"]
                    sentiment = analyzer.polarity_scores(text)
                    text_compound = sentiment["compound"]
                    text_pos = sentiment["pos"]
                    text_neu = sentiment["neu"]
                    text_neg = sentiment["neg"]

                    sentiments.append({
                        "Text": text,
                        "Created": created,
                        "Likes": likes,
                        "Sentiment": sent,
                        "NLTK_Compound" : text_compound,
                        "NLTK_Pos": text_pos,
                        "NLTK_Neu" : text_neu,
                        "NLTK_Neg" : text_neg
                    })
           
                except AttributeError:
                    pass

        df = pd.DataFrame(sentiments)
        df = fix_date(df) 
     
        
        #reorder columns if needed
        #cols = ["date", "text", ...]
        #df = df[cols]
        return df 

    def show_stats(self,df):
        return df.describe()
    
    def sentiment_scaler(self,score):    
        result = 0
        if score >= 0.05:
            result = 1
        elif score < 0.05:
            result = -1
        
        return result

class Blobby:
    def __init__(self):
        pass
    

    
    #---------Probably not needed, but available
    def get_blob(self,text):
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return blob.sentiment

    def get_blob_all(self,text):
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return blob.sentiment.classification, blob.sentiment.p_pos, blob.sentiment.p_neg

    def get_blob_class(self,text):
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return blob.sentiment.classification
    
    def get_blob_pos(self,text):
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return float(blob.sentiment.p_pos)
    
    def get_blob_neg(self,text):
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return float(blob.sentiment.p_neg)
    #----------------------------------------------------
    
    
    def add_blob(self,df, col_name):
        """
        Primary method - takes a dataframe, and the name of the column to analyze (str)
        Appends classification, positive score, and negative score and returns a new dataframe
        
        """
        # Prevents re-training on each iteration - speeds up method by orders of magnitude
        tb = Blobber(analyzer = NaiveBayesAnalyzer())
        
        df = df.copy()
        
        #df = sentiment_cleaner(df)
        df = fix_date(df)
        
        classif = []
        pos = []
        neg = []
    
        for idx, row in enumerate(df.itertuples(index = False)):
            classif.append(tb(df[col_name][idx]).sentiment.classification)
            pos.append(tb(df[col_name][idx]).sentiment.p_pos)
            neg.append(tb(df[col_name][idx]).sentiment.p_neg)
    
        df["Blob Class"] = classif
        df["Blob Pos"]= pos
        df["Blob Neg"] = neg
        return df