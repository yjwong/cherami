#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import tweepy

class BaseClassifier(tweepy.StreamListener):
    def on_status(self, tweet):
        print "Ran on_status"

    def on_error(self, status_code):
        print "Error: " + repr(status_code)
        return False

    def on_status(self, status):
        print status.text

# vim: set ts=4 sw=4 et:
