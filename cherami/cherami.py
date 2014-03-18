#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

from __future__ import division

import sys
import urlparse
import logging

import numpy
import tweepy

from classifier import SVMGlobalClassifier
from classifier import SVMLocalClassifier

from features import FrequencyBasedFeatureSelector
from features import ChiSquareFeatureSelector

from stream import TweetFileStream

from exception import CommandLineException
from exception import ConfigException

logger = logging.getLogger('cherami')

def main():
    import config

    # Initialize logging.
    if config.quiet_mode:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO)

    # Initialize a classifier.
    logger.info('Initializing classifier "{0}"...'.format(
        config.classifier.__name__))

    classifier = config.classifier(config.feature_selector, config.tokenizer,
            **config.classifier_options)
    classifier.train(config.training_sets)

    # Determine the tweet source to use.
    logger.info('Using tweet source "{0}".'.format(config.tweet_source))
    if config.tweet_source == "file":
        logger.info('Loading tweets from "{0}"...'.format(config.tweet_file))
        streamer = TweetFileStream(config.tweet_file, classifier)

        # Realistically speaking, the track parameter never gets used, but is
        # required for the API. So we supply it anyway.
        streamer.filter(track=[config.twitter_track])

        # Perform validation of results.
        results = classifier.get_results()
        for category in config.groundtruth_sets:
            f = open(config.groundtruth_sets[category], 'r')

            # a: Retrieved and Relevant
            # b: Retrieved and Non-Relevant
            # c: Missed and Relevant
            # d: Missed and Non-Relevant
            i = 0
            a = 0
            b = 0
            c = 0
            d = 0

            for line in f:
                if int(line.strip()) == 1: # Relevant
                    if category in results[i]: # Retrieved
                        a += 1
                    else: # Missed
                        c += 1
                elif int(line.strip()) == 0: # Non-Relevant
                    if category in results[i]: # Retrieved
                        b += 1
                    else: # Missed
                        d += 1
                else:
                    raise RuntimeError('Unknown value in ground truth.')

                i += 1

            # Use of numpy.float64 avoids division by zero errors.
            precision = numpy.float64(a) / (a + b)
            recall = numpy.float64(a) / (a + c)

            # Print information.
            print 'Category "{0}": P = {1}, R = {2}, a = {3}, b = {4}, ' \
                'c = {5}, d = {6}'.format(category, precision, recall,
                a, b, c, d)

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
        streamer.filter(track=[config.twitter_track])

    else:
        raise ConfigException("config: invalid tweet source specified")

    return 0

if __name__ == "__main__":
    if config.run_tests:
        classifiers = [ SVMLocalClassifier, SVMGlobalClassifier ]
        feature_selectors = [ FrequencyBasedFeatureSelector, ChiSquareFeatureSelector ]
        max_features_list = [ 8, 16, 32, 64, 128 ]

        # Try ALL the classifiers!
        for classifier_class in classifiers:
            for feature_selector_class in feature_selectors:
                for max_features in max_features_list:
                    print '{0}, {1}, {2}'.format(classifier_class.__name__,
                            feature_selector_class.__name__, max_features)

                    classifier = classifier_class(feature_selector_class, config.tokenizer)
                    classifier.set_max_features(max_features)
                    classifier.train(config.training_sets)

                    logger.info('Loading tweets from "{0}"...'.format(config.tweet_file))
                    streamer = TweetFileStream(config.tweet_file, classifier)

                    # Realistically speaking, the track parameter never gets used, but is
                    # required for the API. So we supply it anyway.
                    streamer.filter(track=[config.twitter_track])

                    # Perform validation of results.
                    results = classifier.get_results()
                    for category in config.groundtruth_sets:
                        f = open(config.groundtruth_sets[category], 'r')

                        # a: Retrieved and Relevant
                        # b: Retrieved and Non-Relevant
                        # c: Missed and Relevant
                        # d: Missed and Non-Relevant
                        i = 0
                        a = 0
                        b = 0
                        c = 0
                        d = 0

                        for line in f:
                            if int(line.strip()) == 1: # Relevant
                                if category in results[i]: # Retrieved
                                    a += 1
                                else: # Missed
                                    c += 1
                            elif int(line.strip()) == 0: # Non-Relevant
                                if category in results[i]: # Retrieved
                                    b += 1
                                else: # Missed
                                    d += 1
                            else:
                                raise RuntimeError('Unknown value in ground truth.')

                            i += 1

                        # Use of numpy.float64 avoids division by zero errors.
                        precision = numpy.float64(a) / (a + b)
                        recall = numpy.float64(a) / (a + c)

                        # Print information.
                        print '{0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(
                                category, precision, recall, a, b, c, d)
                        # print 'Category "{0}": P = {1}, R = {2}, a = {3}, b = {4}, ' \
                        #     'c = {5}, d = {6}'.format(category, precision, recall,
                        #     a, b, c, d)

    else:
        main()

# vim: set ts=4 sw=4 et:
