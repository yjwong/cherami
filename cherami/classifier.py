#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import logging
import json

import tweepy
import numpy
from sklearn import svm

import config

from preprocessor import StopwordRemover
from preprocessor import NLTKTokenizer
from preprocessor import TweetTextFilter
from preprocessor import VocabNormalizer

from exception import ClassifierNotTrainedException

class BaseClassifier(tweepy.StreamListener):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer):
        # Set the feature selector.
        self.feature_selector_class = feature_selector

        # Create the objects to prevent repeated constructions.
        self.text_filter = TweetTextFilter()
        self.remover = StopwordRemover()
        self.remover.build_lists()
        self.tokenizer = tokenizer()
        self.normalizer = VocabNormalizer()
        self.normalizer.build_map()

        # Initialize some state.
        self.training_data = dict()
        self.trained = False
        self.max_features = 64
        self.results = list()

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

    def normalize_term_vector(self, term_vector, features):
        norm = list()
        for feature in features:
            if feature in term_vector:
                # norm.append([feature, term_vector[feature]])
                norm.append([feature, 1])
            else:
                norm.append([feature, 0])
        
        array = numpy.array(norm)
        return array[:,1]

    def set_max_features(self, max_features):
        self.max_features = max_features

    def get_max_features(self):
        return self.max_features

    def set_trained(self, trained):
        self.trained = trained

    def get_trained(self):
        return self.trained

    def get_term_vector(self, status):
        # Filter out links and mentions first.
        if hasattr(status, '__getitem__'):
            text = self.text_filter.filter(status['text'])
        else:
            text = self.text_filter.filter(status.text)

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

    def publish_result(self, status, categories):
        self.print_categories(status, categories)
        self.results.append(categories)

    def get_results(self):
        return self.results

    def print_categories(self, status, categories):
        if not config.quiet_mode:
            if hasattr(status, '__getitem__'):
                status_text = status['text']
            else:
                status_text = status.text

            print u'{0}: ({1})'.format(categories, status_text)
        else:
            print '|'.join(categories)

class SVMLocalClassifier(BaseClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_features = dict()
        self.learning_machines = dict()

        super(SVMLocalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(SVMLocalClassifier, self).train(training_set)
        
        # Local classification: does tweet X belong in category C or not?
        # Since each tweet can belong more than one class, we ask the question
        # for every category C.
        for category in self.training_data:
            self.logger.info('Performing training for category "{0}"...'.format(
                category))

            # Each category has its own set of selected features.
            self.selected_features[category] = \
                    self.feature_selector.get_local_features(category, 0,
                        include_utility=False, max_features=self.max_features)

            term_vectors = list()
            class_labels = list()
            for category_name in self.training_data:
                category_data = self.training_data[category_name]
                for term_vector in category_data:
                    term_vector = self.normalize_term_vector(term_vector,
                            self.selected_features[category])

                    # Create the data required for the SVM classifier.
                    term_vectors.append(term_vector)
                    if category_name == category:
                        class_labels.append(category_name)
                    else:
                        class_labels.append('other')

            # Initialize support vector machine.
            learning_machine = svm.SVC()
            learning_machine.fit(term_vectors, class_labels)
            self.learning_machines[category] = learning_machine

        self.logger.info('Training is complete.')

    def on_status(self, status):
        term_vector = self.get_term_vector(status)
        categories = list()
        for category in self.learning_machines:
            norm_term_vector = self.normalize_term_vector(term_vector,
                    self.selected_features[category])

            learning_machine = self.learning_machines[category]
            prediction = learning_machine.predict(norm_term_vector)
            if prediction[0] != 'other':
                categories.append(prediction[0])

        self.publish_result(status, categories)

class SVMGlobalClassifier(BaseClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer):
        self.logger = logging.getLogger(self.__class__.__name__)
        super(SVMGlobalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(SVMGlobalClassifier, self).train(training_set)

        # Global classification: which category does tweet X belong to?
        self.selected_features = self.feature_selector.get_global_features(
                include_utility=False, max_features=self.max_features)

        term_vectors = list()
        class_labels = list()
        for category_name in self.training_data:
            category_data = self.training_data[category_name]
            for term_vector in category_data:
                term_vector = self.normalize_term_vector(term_vector,
                        self.selected_features)

                # Create the data required for the SVM classifier.
                term_vectors.append(term_vector)
                class_labels.append(category_name)

        # Initialize support vector machine.
        self.learning_machine = svm.SVC()
        self.learning_machine.fit(term_vectors, class_labels)

    def on_status(self, status):
        term_vector = self.get_term_vector(status)
        norm_term_vector = self.normalize_term_vector(term_vector,
                self.selected_features)

        categories = self.learning_machine.predict(norm_term_vector)
        self.publish_result(status, categories)

# vim: set ts=4 sw=4 et:
