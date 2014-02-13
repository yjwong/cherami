#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import tweepy

from preprocessor import StopwordRemover
from preprocessor import SimpleTokenizer
from preprocessor import TweetTextFilter

class BaseClassifier(tweepy.StreamListener):
    def __init__(self):
        self.remover = StopwordRemover()
        self.remover.build_lists()
        super(BaseClassifier, self).__init__()

    def on_error(self, status_code):
        print "Error: " + repr(status_code)
        return False

    def on_status(self, status):
        # Filter out links and mentions first.
        text_filter = TweetTextFilter()
        print status.text
        text = text_filter.filter(status.text)

        # Tokenize the text.
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = self.remover.remove_all(tokens)
        print tokens

# vim: set ts=4 sw=4 et:
