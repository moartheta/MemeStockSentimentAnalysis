import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer


class NLT:
    def __init__(self):
        pass
    
    def update_vader(self):
        nltk.download('vader_lexicon')

    
    def make_sentiment_df(self,df, src = "Reddit"):
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
    
        #adjust later for other API's
        if src.lower() == "reddit":
            #for index,row in df.iterrows():
            #This one is faster if it works
            for index, row in df.itertuples():
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
                     #commenting out unless we use text
#                   if len(text) > 0:
#                     text_compound = sentiment["compound"]
#                     text_pos = sentiment["pos"]
#                     text_neu = sentiment["neu"]
#                     text_neg = sentiment["neg"]
#                   else:
#                     text_compound = np.nan
#                     text_pos = np.nan
#                     text_neu = np.nan
#                     text_neg = np.nan

                    sentiments.append({
                        "Title": title,
                        "Upvote_Ratio": upvote,
                        "Reddit_Title_Compound": title_compound,
                        "Reddit_Title_Positive": title_pos,
                        "Reddit_Title_Negative": title_neg,
                        "Reddit_Title_Neutral" : title_neu,
#                     "Text_Compound" : text_compound,
#                     "Text_Positive" : text_pos,
#                     "Text_Negative" : text_neg,
#                     "Text_Neutral" : text_neu
                    })
                except AttributeError:
                    pass
        
        df = pd.DataFrame(sentiments)
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

    def add_blob(self,df):
        tb = Blobber(analyzer = NaiveBayesAnalyzer())
        df = df.copy()

        classif = []
        pos = []
        neg = []
    
        for idx, row in enumerate(df.itertuples(index = False)):
            classif.append(tb(df["Title"][idx]).sentiment.classification)
            pos.append(tb(df["Title"][idx]).sentiment.p_pos)
            neg.append(tb(df["Title"][idx]).sentiment.p_neg)
    
        df["Blob Class"] = classif
        df["Blob Pos"]= pos
        df["Blob Neg"] = neg
        return df