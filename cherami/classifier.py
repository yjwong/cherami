#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import json
import tweepy
from collections import OrderedDict

from preprocessor import StopwordRemover
from preprocessor import SimpleTokenizer
from preprocessor import TweetTextFilter
from preprocessor import VocabNormalizer

from exception import ClassifierNotTrainedException

class BaseClassifier(tweepy.StreamListener):
    def __init__(self):
        # Create the objects to prevent repeated constructions.
        self.text_filter = TweetTextFilter()
        self.remover = StopwordRemover()
        self.remover.build_lists()
        self.tokenizer = SimpleTokenizer()
        self.normalizer = VocabNormalizer()
        self.normalizer.build_map()

        # Initialize some state.
        self.trained = False
        self.training_data = list()

        super(BaseClassifier, self).__init__()

    def train(self, training_file):
        # Create some classes first 
        text_filter = TweetTextFilter()

        # Start reading from the training set.
        f = open(training_file, 'r')
        for line in f:
            status = json.loads(line)

            # Filter out links and mentions first.
            text = self.text_filter.filter(status['text'])

            # Tokenize the text.
            tokens = self.tokenizer.tokenize(text)
            tokens = self.remover.remove_all(tokens)

            # Normalize the vocabulary.
            tokens = self.normalizer.normalize(tokens)

            # Create the term vector.
            term_vector = dict()
            for token in tokens:
                if token in term_vector:
                    term_vector[token] += 1
                else:
                    term_vector[token] = 1
            
            self.training_data.append(term_vector)

        self.trained = True

    def on_error(self, status_code):
        print "Error: " + repr(status_code)
        return False

    def on_status(self, status):
        if self.trained is False:
            raise ClassifierNotTrainedException('Classifier must be trained '
                    'before use.')

        # Filter out links and mentions first.
        text = self.text_filter.filter(status.text)

        # Tokenize the text.
        tokens = self.tokenizer.tokenize(text)
        tokens = self.remover.remove_all(tokens)

        # Normalize the vocabulary.
        tokens = self.normalizer.normalize(tokens)

class GlobalClassifier(BaseClassifier):
    def __init__(self):
        self.df_threshold = 50
        super(GlobalClassifier, self).__init__()

    def compute_df(self, term):
        """
        Computes the document frequency for a term.
        This is the number of documents containing a particular term.
        """
        term = term.lower()
        df = 0

        for term_vector in self.training_data:
            if term in term_vector:
                df += 1
        
        return df

    def compute_df_all(self, sort=False, thresholding=True):
        """
        Computes the document frequency for all terms.

        If sort is True, this returns a sorted list of tuples. Otherwise, a
        dict will be returned.

        If thresholding is True, document frequency thresholding is used. To
        set the threshold, use set_df_threshold. This parameter is ignored if
        sort is False.
        """
        df_map = dict()
        for term_vector in self.training_data:
            for term in term_vector:
                if term not in df_map:
                    df_map[term] = self.compute_df(term)

        if sort:
            sorted_df_map = OrderedDict(
                    sorted(df_map.items(), key=lambda t: t[1], reverse=True))
            return [(item, sorted_df_map[item]) for item in sorted_df_map
                    if sorted_df_map[item] > self.df_threshold]

        else:
            return df_map

    def set_df_threshold(self, threshold):
        self.df_threshold = threshold

# vim: set ts=4 sw=4 et:
