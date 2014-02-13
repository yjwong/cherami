#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import tweepy

from preprocessor import StopwordRemover
from preprocessor import SimpleTokenizer
from preprocessor import TweetTextFilter
from preprocessor import VocabNormalizer

class BaseClassifier(tweepy.StreamListener):
    def __init__(self):
        # Create the objects to prevent repeated constructions.
        self.remover = StopwordRemover()
        self.remover.build_lists()
        self.tokenizer = SimpleTokenizer()
        self.normalizer = VocabNormalizer()
        self.normalizer.build_map()

        super(BaseClassifier, self).__init__()

    def on_error(self, status_code):
        print "Error: " + repr(status_code)
        return False

    def on_status(self, status):
        # Filter out links and mentions first.
        text_filter = TweetTextFilter()
        text = text_filter.filter(status.text)

        # Tokenize the text.
        tokens = self.tokenizer.tokenize(text)
        tokens = self.remover.remove_all(tokens)

        # Normalize the vocabulary.
        tokens = self.normalizer.normalize(tokens)

# vim: set ts=4 sw=4 et:
