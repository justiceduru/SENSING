import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import csv
import time
import pandas as pd
import csv
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators


api_key ='' #request access from AlphaVantage

class TwitterClient(object):
    #  Generic Twitter Class for sentiment analysis.

    def __init__(self):
        # Class constructor or initialization method.

        # keys and tokens from the Twitter Dev Console
        consumer_key = ''
        consumer_secret = ''
        access_token = ''
        access_token_secret = ''

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        #  Utility function to clean tweet text by removing links, special characters using simple regex statements.
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        # Utility function to classify sentiment of passed tweet using textblob's sentiment method

        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'


    def get_tweets(self, query, geocode, since_id, count=10):
        #  Main function to fetch tweets and parse them.

        # empty list to store parsed tweets
        tweets = []
        tweet_text=[]

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q=query, count=count, geocode=geocode, since_id=since_id)

            # parsing tweets one by one
            for tweet in fetched_tweets:

                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                # saving timestamp of tweet
                
                # saving id of tweet
                parsed_tweet['id'] = tweet.id

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet['text'] not in tweet_text:
                        tweets.append(parsed_tweet)
                        tweet_text.append(parsed_tweet['text'])

                else:
                    tweets.append(parsed_tweet)
                    tweet_text.append(parsed_tweet['text'])

            # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    oldtweets=[]
    start_time = time.time()
    upload_timer = time.time()
    print(start_time)
    last_id=0
    loop = 0
    csvtitle = 'Tweets_1.csv'
    title_iterator = 1

    while True:

        # calling function to get most recent tweets
        tweetbatch = api.get_tweets(query= lang:en point_radius:[-121.8863 37.3382 1.2km], geocode="-121.8863, 37.3382, 1.2km", since_id=last_id, count=1000)
        ts = TimeSeries(key=api_key, output_format='pandas')
        ti = TechIndicators(key=api_key, output_format='pandas')
        period = 60
        GOOGLdata_ts, GOOGLmeta_data_ts = ts.get_intraday(symbol='GOOGL', interval='5min', outputsize='full')
        print(GOOGLmeta_data_ts)

        GOOGLdata_ti, GOOGLmeta_data_ti = ti.get_sma(symbol='GOOGL', interval='5min',
                                            time_period=period, series_type='close')
        GOOGLdf1 = GOOGLdata_ti
        GOOGLdf2 = GOOGLdata_ts['4. close'].iloc[period-1::]

        EBAYdata_ts, EBAYmeta_data_ts = ts.get_intraday(symbol='EBAY', interval='5min', outputsize='full')
        print(EBAYmeta_data_ts)

        EBAYdata_ti, EBAYmeta_data_ti = ti.get_sma(symbol='EBAY', interval='5min',
                                            time_period=period, series_type='close')
        EBAYdf1 = EBAYdata_ti
        EBAYdf2 = EBAYdata_ts['4. close'].iloc[period-1::]



        GOOGLdf2.index = GOOGLdf1.index
        EBAYdf2.index = EBAYdf1.index

        GOOGLtotal_df = pd.concat([GOOGLdf1, GOOGLdf2], axis=1)
        EBAYtotal_df = pd.concat([EBAYdf1, EBAYdf2], axis=1)

        GOOGLtotal_df.to_csv('GOOGLStock.csv', encoding='utf-8')
        EBAYtotal_df.to_csv('EBAYStock.csv', encoding='utf-8')
        
        
        if tweetbatch is not None: # check that there are new tweets
            freshtweets = [x for x in tweetbatch if x not in oldtweets]  # remove any that matched the last sample
            freshtweets.reverse()  # make chronological
            oldtweets = tweetbatch  # reassign the most recent batch of tweets

            # store sentiment, timestamp and text of each tweet in newline of csv
            with open(csvtitle, 'a') as csvFile:
                writer = csv.writer(csvFile)

                # Calculate sentiment for each tweet
                for tweet in freshtweets:
                    sentiment = tweet['sentiment']
                    timestamp = tweet[0]['timestamp']
                    text = tweet['text']
                    row = [loop, timestamp, sentiment, text]
                    writer.writerow(row)
                csvFile.close()

            # get last ID
            print(len(freshtweets))
            if len(freshtweets) > 0:
                last_id = freshtweets[-1]['id']

        # if 24 hours has passed upload the file and change title
        twentyfourhours = 60*60*24
        upload_time_passed = time.time() - upload_timer
        if upload_time_passed > twentyfourhours:
            upload_timer = time.time()  # reset to 0 seconds
            csvtitle = 'Tweets_' + str(title_iterator) + '.csv'  # next title
            title_iterator = title_iterator + 1

        # wait 5 minutes and repeat
        time.sleep(300)
        loop = loop + 1


if __name__ == "__main__":
    main()
