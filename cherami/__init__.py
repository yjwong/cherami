#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details.

import json

"""
Cher Ami: A Tweet Classifier
"""
__version__ = "0.1"
__author__ = "Wong Yong Jie"
__license__ = "Apache"

def classify_tweets(tweets, training_tweets, training_labels, classifier_class,
        feature_selector_class, tokenizer_class, stopword_sources,
        vocab_map_file):
    if len(training_tweets) != len(training_labels):
        raise StandardError("Number of training tweets does not match number "
                "of training labels.")

    classifier = classifier_class(feature_selector_class, tokenizer_class,
	    stopword_sources, vocab_map_file)
    classifier.train(training_tweets, training_labels)
    
    # Convert tweet to status.
    for tweet in tweets:
        tweet = json.loads(unicode(tweet, "ISO-8859-1"))
        classifier.on_status(tweet)

    for result in classifier.get_results():
        print result

# vim: set ts=4 sw=4 et:
