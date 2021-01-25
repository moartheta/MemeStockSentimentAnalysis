# Can you really drive a stock price with social media hype?
# The rise of Robinhood: 

By: Camillo D'Orazio, Todd Shevlin, Daniel Singer, Gregory Terrinoni - January 2021

![Social Media](https://i1.wp.com/thedatascientist.com/wp-content/uploads/2018/10/sentiment-analysis.png)<sup>1<sup>

# Overview 

The scope of the project is to look at some momemtum stocks that are most talked about on social media platforms, and more specifically, are hyped by millenials. Perform sentiment analysis on 5 momentum stocks - Tesla, GameStop, PLTR, PLUG, and NIO. <sup>2<sup> 

We will use various tools and machine learning models to predict prices using closing prices of momentum stocks and perform sentiment analysis on these 5 highly touted stocks on social media platforms.

# Hypothesis
## Does social media chatter matter for stock prices?

*"Is what people say really a driver for stock price movements?"*

Our hypothesis for this project is to determine whether social meda chatter has an impact on the price movement of certain stocks regardless of any financial fundemental analysis.

# Model Summary

#### There were three total models used for the project:
1) NLTK - A well-known Natural Language Processing package used for sentiment analysis.
2) TextBlob - an alternative sentiment analysis tool, used in conjunction with the NLTK model.
3) The LSTM (Long Short Term Memory Recurrent Neural Network) model via Tensorflow/Keras, as it is one of the strongest predictive models we have used.

If there had been additional time, we also considered using a Logistic Regression model.  

# Data Cleanup & Model Training

*Describe the exploration and cleanup process.
Discuss any problems that arose with preparing the data or training the model that you didn't anticipate.
Discuss the overall training process and highlight anything of interest with the training process: Cloud resources used, training time required, issues with training.*
#### Obtaining And Structuring The Data
1) Obtaining the data from various API's was challenging.  Some API's were not working correctly, others severely limited the available data.  In the end we utilized data from Reddit, StockTwits, and Alpaca.
2) There were some dataframe manipulation challenges, assuring indexes and dates were in the correct format, column headings and data types for the rows were consistent for concatenating and merging, and getting to the "final dataframes".  However, it was just a matter of going through the process, and some trial and error.

#### Model Development
1) We separated the models, classes, and functions into separate files.  This made code maintenance and troubleshooting much easier, not to mention cutting down on the total code needed.
2) On the sentiment models, the largest challenge was getting the TextBlobs to run faster.  By default it retrains every time it runs, which was causing major performance issues (about 20 minutes for every 1000 rows).  After searching, StackOverflow presented a solution to prevent re-training, and improved the performance by orders of magnitude.
3) The largest challenge we ran into on the LSTM model was using multiple features.  This took a fair amount of searching and fine tuning.  Other than this, it was pretty standard model development.   


# Model Evaluation

*Discuss the techniques you used to evaluate the model performance.*

# Analysis Results / Observations

*Discuss your findings. Was the model sufficient for the predictive task? If not, why not? What inferences or general conclusions can you draw from your model performance?*  

# Postmortem

*Discuss any difficulties that arose, and how you dealt with them.
Discuss any additional questions or problems that came up but you didn't have time to answer: What would you research next if you had two more weeks?*
There was definitely some user error at times.  Referencing the incorrect dataframe, overwriting one class with another class, or running/outputting to the incorrect file.
Overall, we probably spent too much time getting the data into the correct format for the models.  This should improve with experience and practice.
The API limitations caused us to switch over to CSVs for input, which worked fine for the build.  In a production environment we would switch out the CSV inputs to API calls.
With additional time and data availability, we would use more data to train the model to see if it improves, add additional features, and increase the comparisons to determine the strength of the effect of different features on the model.  

## Datasets to be used
- Institutional vs. Retail Investor (Non-Insitutional) data
- Reddit API
- Quandl API
- Alpaca API
- Twitter API
- News API
- Discord API
- Telegram API
- StockTwits API (Beautiful Soup as a backstop)








____________________________________________________________________________________________________________________
<sup>1 Emoji images - https://i1.wp.com/thedatascientist.com/wp-content/uploads/2018/10/sentiment-analysis.png

<sup>2 Duggan, W. (2021, January 8) *10 Momentum Stocks Millennials Love in 2021* U.S. News and World Report https://money.usnews.com/investing/stock-market-news/slideshows/momentum-stocks-millennials-are-buying?slide=7


<sup>3 Moeller, M. (2018, February 26) *Ethereum Competitors: Guide to the Alternative Smart Contract Platforms* Blockonomi.com https://blockonomi.com/ethereum-competitors/

<sup>4 Dughi, P. (2018, February 3) *A simple explanation of how blockchain works* Mission.org https://medium.com/the-mission/a-simple-explanation-on-how-blockchain-works-e52f75da6e9a#:~:text=Blockchain%20is%20the%20technology%20the%20underpins%20digital%20currency,a%20%E2%80%9Cdigital%20ledger%E2%80%9D%20stored%20in%20a%20distributed%20network.

<sup>5 Reiff, N. (2020, June 16) *Bitcoin vs. Ethereum: What's the Difference?* Investopedia https://www.investopedia.com/articles/investing/031416/bitcoin-vs-ethereum-driven-different-purposes.asp.

<sup>6 YouTube video ... Justin to document

<sup>7 Denominations of Ether source: https://nagritech.com/wp-content/uploads/2020/04/Screenshot-22.png

<sup>8 

## Perform machine learning, produce graphs, make interactive via a Dashboard
- Good example .... http://www.sentdex.com/financial-analysis/?i=GME&tf=all&style=overlay

Below is another format for including web images in this document.

<img src='https://www.fxcintel.com/wp-content/uploads/cust_acquiring-1.png' width='400'><sup>15<sup>

