#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import json

import tweepy
from util import ISO8601toTwitterDate

class TweetFileStream(tweepy.Stream):
    def __init__(self, path, listener, **options):
        self.file = open(path)
        self.listener = listener

    def filter(self, follow=None, track=None, async=False, locations=None,
               count=None, stall_warnings=False, languages=None,
               encoding="utf8"):
        for line in self.file:
            # TODO: Satisfy the parameters given
            status = json.loads(line)

            # Normalize the date to the standard Twitter format.
            status["created_at"] = ISO8601toTwitterDate(status["created_at"]["$date"])
            status["user"]["created_at"] = ISO8601toTwitterDate(status["user"]["created_at"]["$date"])

            # Send for processing.
            self.listener.on_data(json.dumps(status))

# vim: set ts=4 sw=4 et:
