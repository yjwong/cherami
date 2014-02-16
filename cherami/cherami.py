#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import sys
import urlparse
import logging

import tweepy
import config

from classifier import GlobalClassifier
from classifier import LocalClassifier

from stream import TweetFileStream

from features import FrequencyBasedFeatureSelector
from features import ChiSquareFeatureSelector

from exception import CommandLineException
from exception import ConfigException

logger = logging.getLogger('cherami')

def main():
    # Initialize logging.
    logging.basicConfig(level=logging.INFO)

    # Initialize a classifier.
    logger.info('Initializing classifier with mode "{0}"...'.format(
        config.classifier_mode))

    if config.classifier_mode == 'global':
        classifier = GlobalClassifier(FrequencyBasedFeatureSelector)
        classifier.train(config.training_sets)

    elif config.classifier_mode == 'local':
        classifier = LocalClassifier(ChiSquareFeatureSelector)
        # classifier = LocalClassifier(FrequencyBasedFeatureSelector)
        classifier.train(config.training_sets)

    else:
        raise NotImplementedError('Classifier mode "{0}" not implemented.'.
                format(config.classifier_mode))

    # Determine the tweet source to use.
    logger.info('Using tweet source "{0}".'.format(config.tweet_source))
    if config.tweet_source == "file":
        logger.info('Loading tweets from "{0}"...'.format(config.tweet_file))
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

        logger.info('Retrieving link: {0}'.format(sys.argv[1]))
        api = tweepy.API(auth, parser=tweepy.parsers.RawParser())
        result = api.get_status(status_id)
        classifier.on_data(result)

    elif config.tweet_source == "twitter":
        auth = tweepy.OAuthHandler(config.oauth_consumer_key, config.oauth_consumer_secret)
        auth.set_access_token(config.oauth_token_key, config.oauth_token_secret)

        logger.info('Connecting to Twitter Streaming API...')
        streamer = tweepy.Stream(auth, classifier)
        streamer.filter(track=["twitter"])

    else:
        raise ConfigException("config: invalid tweet source specified")

    return 0

if __name__ == "__main__":
    main()

# vim: set ts=4 sw=4 et:
