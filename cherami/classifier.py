#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import tweepy

from preprocessor import StopwordRemover
from preprocessor import SimpleTokenizer

class BaseClassifier(tweepy.StreamListener):
    def __init__(self):
        self.remover = StopwordRemover()
        self.remover.build_lists()
        super(BaseClassifier, self).__init__()

    def on_error(self, status_code):
        print "Error: " + repr(status_code)
        return False

    def on_status(self, status):
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(status.text)
        tokens = self.remover.remove(tokens)
        tokens = self.remover.remove_mentions(tokens)
        print(tokens)

# vim: set ts=4 sw=4 et:
