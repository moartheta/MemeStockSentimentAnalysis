import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

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


def get_blob(text):
    blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    return blob.sentiment

def get_blob_class(text):
    blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    return blob.sentiment.classification
    
def get_blob_pos(text):
    blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    return float(blob.sentiment.p_pos)
    
def get_blob_neg(text):
    blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    return float(blob.sentiment.p_neg)

def add_blob(df):
    blobbies = []
    
    df = df.copy()
    
    #classif = [get_blob_class(x) for x in df["Title"]]
    #pos = [get_blob_pos(x) for x in df["Title"]]
    #neg = [get_blob_neg(x) for x in df["Title"]]
    
    #print(classif)
    #print(pos)
    #print(neg)
    
    
    for index,row in df.iterrows():
        s = get_blob(row["Title"])
        classif = s.classification
        pos = s.p_pos
        neg = s.p_neg
                
        blobbies.append({
            "Blob Class" : classif,
            "Blob Pos" : pos,
            "Blob Neg" : neg
        })
        print(classif)
        
    df_blob = pd.DataFrame(blobbies)
    return df_blob
                
#     df["Blob_Class"] = ""
#     df["Blob_Pos"] = 0.0
#     df["Blob_Neg"] = 0.0
    
#     for item in df["Title"]:
#         print(item)
#         s = get_blob(item)
#         #df["Blob_Class"].append(get_blob_class(item))
#         #df["Blob_Pos"].append(get_blob_pos(item))
#         #df["Blob_Neg"].append(get_blob_neg(item))
#         df["Blob_Class"].append(s.classification)
#         df["Blob_Pos"].append(s.p_pos)
#         df["Blob_Neg"].append(s.p_neg)
     
#     return df
# #     for row, index in df.iterrows():
# #         classif.append(list(get_blob_class(row[0])))
# #         pos.append(get_blob_pos(row["Title"]))
# #         neg.append(get_blob_neg(row["Title"]))
        
# #     df["Blob_Class"] = classif
# #     df["Blob_Pos"] = pos
# #     df["Blob_Neg"] = neg
    
# #    return df
# #        df["Blob_Class"] = map(lambda x: get_blob_class(x), df["Title"])
#  #       df["Blob_Pos"] = list(map(get_blob_pos,df["Title"]))
#   #      df["Blob_Neg"] = [x: for item in list item == get_blob_neg(x)]