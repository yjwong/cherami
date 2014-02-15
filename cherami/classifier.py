#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import logging
import json

import tweepy

from preprocessor import StopwordRemover
from preprocessor import SimpleTokenizer
from preprocessor import TweetTextFilter
from preprocessor import VocabNormalizer

from exception import ClassifierNotTrainedException

class BaseClassifier(tweepy.StreamListener):
    def __init__(self, feature_selector):
        # Set the feature selector.
        self.feature_selector_class = feature_selector

        # Create the objects to prevent repeated constructions.
        self.text_filter = TweetTextFilter()
        self.remover = StopwordRemover()
        self.remover.build_lists()
        self.tokenizer = SimpleTokenizer()
        self.normalizer = VocabNormalizer()
        self.normalizer.build_map()

        # Initialize some state.
        self.training_data = dict()
        self.trained = False

        super(BaseClassifier, self).__init__()

    def train(self, training_sets):
        for set_name in training_sets:
            training_file = training_sets[set_name]
            set_data = list()

            self.logger.info('Reading training set "{0}" ({1})...'.format(
                set_name, training_file))

            # Read JSON from the set.
            f = open(training_file, 'r')
            for line in f:
                status = json.loads(line)
                term_vector = self.get_term_vector(status)
                set_data.append(term_vector)

            self.training_data[set_name] = set_data

        self.logger.info('Reading training sets complete.')
        self.set_trained(True)

        # Create the feature selector.
        self.feature_selector = self.feature_selector_class(self.training_data)

    def set_trained(self, trained):
        self.trained = trained

    def get_trained(self):
        return self.trained

    def get_term_vector(self, status):
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

        return term_vector

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

class LocalClassifier(BaseClassifier):
    def __init__(self, feature_selector):
        self.logger = logging.getLogger(self.__class__.__name__)
        super(LocalClassifier, self).__init__(feature_selector)

    def train(self, training_set):
        super(LocalClassifier, self).train(training_set)
        
        # Local classification: does tweet X belong in category C or not?
        print self.feature_selector.get_local_features('nus1', 0, include_utility=True)

class GlobalClassifier(BaseClassifier):
    def __init__(self, feature_selector):
        self.logger = logging.getLogger(self.__class__.__name__)
        super(GlobalClassifier, self).__init__(feature_selector)

    def train(self, training_set):
        super(GlobalClassifier, self).train(training_set)

        # Global classification: which category does tweet X belong to?
        print self.feature_selector.get_global_features(use_max=False, include_utility=True)

# vim: set ts=4 sw=4 et:
