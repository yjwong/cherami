#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import sys
import urlparse

import tweepy
import config

from classifier import BaseClassifier
from stream import TweetFileStream
from exception import CommandLineException
from exception import ConfigException

def main():
    # Initialize a classifier.
    classifier = BaseClassifier()

    # Determine the tweet source to use.
    if config.tweet_source == "file":
        streamer = TweetFileStream(config.tweet_file, classifier)
        streamer.filter(track=["twitter"])
    
    elif config.tweet_source == "link":
        auth = tweepy.OAuthHandler(config.oauth_consumer_key, config.oauth_consumer_secret)
        auth.set_access_token(config.oauth_token_key, config.oauth_token_secret)

        # The 'link' option requires a argument - the link itself.
        if len(sys.argv) != 2:
            raise CommandLineException("main: the \"link\" tweet source "
                    "requires an argument")

        # Parse the URL.
        url_info = urlparse.urlsplit(sys.argv[1])
        status_id = url_info.path.rsplit("/", 1)[1]

        api = tweepy.API(auth, parser=tweepy.parsers.RawParser())
        result = api.get_status(status_id)
        classifier.on_data(result)

    elif config.tweet_source == "twitter" or \
         config.tweet_source == "link":
        auth = tweepy.OAuthHandler(config.oauth_consumer_key, config.oauth_consumer_secret)
        auth.set_access_token(config.oauth_token_key, config.oauth_token_secret)

        streamer = tweepy.Stream(auth, classifier)
        streamer.filter(track=["twitter"])

    else:
        raise ConfigException("config: invalid tweet source specified")

    return 0

if __name__ == "__main__":
    main()

# vim: set ts=4 sw=4 et:
