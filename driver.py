#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 2
# See LICENSE for details

import cherami
from cherami.classifier import SVMGlobalClassifier
from cherami.features import FrequencyBasedFeatureSelector
from cherami.preprocessor import NLTKTokenizer

if __name__ == "__main__":
    training_tweets = list()
    training_labels = list()
    tweets = list()

    training_tweets_file = open("./dataset/training/tweets_train.txt")
    for tweet in training_tweets_file:
        training_tweets.append(tweet)

    training_labels_file = open("./dataset/training/label_train.txt")
    for line in training_labels_file:
        label, sentiment, tweet_id = line.split(',')
        training_labels.append(label)

    tweets_file = open("./dataset/testing/tweets_test.txt")
    for line in tweets_file:
        tweets.append(line)
    
    stopword_sources = [
        'nltk:english',
        'file:./dataset/stopwords.smartdatacollective.com.txt'
    ]

    vocab_map_file = './dataset/normalization_map.emnlp.txt'

    test = cherami.classify_tweets(tweets=tweets,
            training_tweets=training_tweets,
            training_labels=training_labels,
            classifier_class=SVMGlobalClassifier,
            feature_selector_class=FrequencyBasedFeatureSelector,
            tokenizer_class=NLTKTokenizer,
            stopword_sources=stopword_sources,
            vocab_map_file=vocab_map_file)

# vim: set ts=4 sw=4 et:
