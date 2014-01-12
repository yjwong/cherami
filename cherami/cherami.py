#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import tweepy
import config

def main():
    auth = tweepy.OAuthHandler(config.oauth_consumer_key, config.oauth_consumer_secret)
    auth.set_access_token(config.oauth_token_key, config.oauth_token_secret)

    class StreamListener(tweepy.StreamListener):
        def on_status(self, tweet):
            print "Ran on_status"

        def on_error(self, status_code):
            print "Error: " + repr(status_code)
            return False
        
        def on_data(self, data):
            print data

    listener = StreamListener()
    streamer = tweepy.Stream(auth, listener)
    streamer.filter(track=["twitter"])

    return 0

if __name__ == "__main__":
    main()

# vim: set ts=4 sw=4 et:
