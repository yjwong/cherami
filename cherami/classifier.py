#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import logging
import json
import warnings

import tweepy
import numpy

with warnings.catch_warnings():
    from sklearn import svm
from sklearn import neighbors
from sklearn import tree

import config

from preprocessor import StopwordRemover
from preprocessor import NLTKTokenizer
from preprocessor import TweetTextFilter
from preprocessor import VocabNormalizer

from exception import ClassifierNotTrainedException

class BaseClassifier(tweepy.StreamListener):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        # Set the feature selector.
        self.feature_selector_class = feature_selector

        # Create the objects to prevent repeated constructions.
        self.text_filter = TweetTextFilter()
        self.remover = StopwordRemover()
        self.remover.build_lists()
        self.tokenizer = tokenizer()
        self.normalizer = VocabNormalizer()
        self.normalizer.build_map()
        self.max_features = config.max_features

        # Initialize some state.
        self.training_data = dict()
        self.trained = False
        self.results = list()

        super(BaseClassifier, self).__init__()

    def train(self, training_sets):
        # Don't allow retraining.
        if self.trained:
            raise RuntimeError('Classifier is already trained')

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

    def get_data_count(self):
        data_count = 0

        for category_name in self.training_data:
            category_data = self.training_data[category_name]
            data_count += len(category_data)

        return data_count

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

class LocalClassifier(BaseClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_features = dict()
        self.term_vectors = dict()
        self.class_labels = dict()

        super(LocalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(LocalClassifier, self).train(training_set)

        # Local classification: does tweet X belong in category C or not?
        # Since each tweet can belong more than one class, we ask the question
        # for every category C.
        data_count = self.get_data_count()
        for category in self.training_data:
            self.selected_features[category] = \
                    self.feature_selector.get_local_features(category, 0,
                        include_utility=False, max_features=self.max_features)

            # Create the list of term vectors.
            term_vectors = numpy.empty([data_count, self.max_features])
            term_vector_idx = 0
            class_labels = [None] * data_count

            # For each item in the category, classify into the category and
            # 'others'.
            for category_name, category_data in self.training_data.iteritems():
                for term_vector in category_data:
                    term_vector = self.normalize_term_vector(term_vector,
                            self.selected_features[category])

                    # Create the data required for the SVM classifier.
                    term_vectors[term_vector_idx] = term_vector
                    if category_name == category:
                        class_labels[term_vector_idx] = category_name
                    else:
                        class_labels[term_vector_idx] = 'other'
                    
                    term_vector_idx += 1

            self.term_vectors[category] = term_vectors
            self.class_labels[category] = class_labels

    def get_term_vectors(self, category):
        return self.term_vectors[category]

    def get_class_labels(self, category):
        return self.class_labels[category]

class GlobalClassifier(BaseClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_features = dict()
        self.term_vectors = dict()
        self.class_labels = dict()

        super(GlobalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(GlobalClassifier, self).train(training_set)

        # Global classification: which category does tweet X belong to?
        self.selected_features = self.feature_selector.get_global_features(
                include_utility=False, max_features=self.max_features)

        data_count = self.get_data_count()
        term_vectors = numpy.empty([data_count, self.max_features])
        term_vector_idx = 0
        class_labels = [None] * data_count

        for category_name in self.training_data:
            category_data = self.training_data[category_name]
            for term_vector in category_data:
                term_vector = self.normalize_term_vector(term_vector,
                        self.selected_features)

                # Create the data required for the SVM classifier.
                term_vectors[term_vector_idx] = term_vector
                class_labels[term_vector_idx] = category_name
                term_vector_idx += 1

        self.term_vectors = term_vectors
        self.class_labels = class_labels

    def get_term_vectors(self):
        return self.term_vectors
    
    def get_class_labels(self):
        return self.class_labels

class SVMLocalClassifier(LocalClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
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

            # Initialize support vector machine.
            learning_machine = svm.SVC()
            learning_machine.fit(self.get_term_vectors(category),
                    self.get_class_labels(category))
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

class SVMGlobalClassifier(GlobalClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        super(SVMGlobalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(SVMGlobalClassifier, self).train(training_set)

        # Initialize support vector machine.
        self.learning_machine = svm.SVC()
        self.learning_machine.fit(self.get_term_vectors(), 
                self.get_class_labels())

    def on_status(self, status):
        term_vector = self.get_term_vector(status)
        norm_term_vector = self.normalize_term_vector(term_vector,
                self.selected_features)

        categories = self.learning_machine.predict(norm_term_vector)
        self.publish_result(status, categories)

class kNNLocalClassifier(LocalClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.learning_machines = dict()

        super(kNNLocalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(kNNLocalClassifier, self).train(training_set)
        
        # Local classification: does tweet X belong in category C or not?
        # Since each tweet can belong more than one class, we ask the question
        # for every category C.
        for category in self.training_data:
            self.logger.info('Performing training for category "{0}"...'.format(
                category))

            # Initialize support vector machine.
            learning_machine = neighbors.KNeighborsClassifier(512,
                    weights='uniform')
            learning_machine.fit(self.get_term_vectors(category),
                    self.get_class_labels(category))
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

class kNNGlobalClassifier(GlobalClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        super(kNNGlobalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(kNNGlobalClassifier, self).train(training_set)
        
        # Initialize kNN classifier.
        self.learning_machine = neighbors.KNeighborsClassifier(256,
                weights='uniform')
        self.learning_machine.fit(self.get_term_vectors(),
                self.get_class_labels())

    def on_status(self, status):
        term_vector = self.get_term_vector(status)
        norm_term_vector = self.normalize_term_vector(term_vector,
                self.selected_features)
        
        categories = self.learning_machine.predict(norm_term_vector)
        self.publish_result(status, categories)

class DecisionTreeLocalClassifier(LocalClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.learning_machines = dict()

        super(DecisionTreeLocalClassifier, self).__init__(feature_selector,
                tokenizer)

    def train(self, training_set):
        super(DecisionTreeLocalClassifier, self).train(training_set)
        
        # Local classification: does tweet X belong in category C or not?
        # Since each tweet can belong more than one class, we ask the question
        # for every category C.
        for category in self.training_data:
            self.logger.info('Performing training for category "{0}"...'.format(
                category))

            # Initialize support vector machine.
            learning_machine = tree.DecisionTreeClassifier()
            learning_machine.fit(self.get_term_vectors(category),
                    self.get_class_labels(category))
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

class DecisionTreeGlobalClassifier(GlobalClassifier):
    def __init__(self, feature_selector, tokenizer=NLTKTokenizer, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        super(DecisionTreeGlobalClassifier, self).__init__(feature_selector, tokenizer)

    def train(self, training_set):
        super(DecisionTreeGlobalClassifier, self).train(training_set)
        
        # Initialize decision tree classifier.
        self.learning_machine = tree.DecisionTreeClassifier()
        self.learning_machine.fit(self.get_term_vectors(),
                self.get_class_labels())

    def on_status(self, status):
        term_vector = self.get_term_vector(status)
        norm_term_vector = self.normalize_term_vector(term_vector,
                self.selected_features)
        
        categories = self.learning_machine.predict(norm_term_vector)
        self.publish_result(status, categories)

# vim: set ts=4 sw=4 et:
