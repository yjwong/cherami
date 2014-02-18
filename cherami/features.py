#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

from __future__ import division

import sys
from collections import OrderedDict
from exception import UnknownCategoryException

class FeatureSelector(object):
    def __init__(self, training_data):
        self.training_data = training_data

    def set_global_threshold(self, global_threshold):
        self.global_threshold = global_threshold

    def get_global_threshold(self):
        return self.global_threshold

    def get_global_features(self, include_utility=False, max_features=-1):
        raise NotImplementedError('get_global_features not implemented.')

    def get_local_features(self, category, threshold, include_utility=False,
            max_features=-1):
        raise NotImplementedError('get_local_features not implemented.')

    def get_terms(self):
        """ Returns a set of all terms across all training sets. """
        terms = set()
        for set_name in self.training_data:
            set_data = self.training_data[set_name]
            for term_vector in set_data:
                for term in term_vector:
                    terms.add(term)

        return terms

    def get_category_terms(self, category):
        """ Returns a set of terms in one training set. """
        terms = set()
        category_data = self.training_data[category]
        for term_vector in category_data:
            for term in term_vector:
                terms.add(term)

        return terms

class ChiSquareFeatureSelector(FeatureSelector):
    def __init__(self, training_data):
        self.global_threshold = 0
        super(ChiSquareFeatureSelector, self).__init__(training_data)

    def get_global_features(self, include_utility=False, max_features=-1):
        global_chisquares = dict()
        for category_name in self.training_data:
            chisquares = self.compute_chisquare_all(category_name)
            for term in chisquares:
                if term not in global_chisquares:
                    global_chisquares[term] = chisquares[term]
                else:
                    if chisquares[term] > global_chisquares[term]:
                        global_chisquares[term] = chisquares[term]

        chisquares = self.sort_chisquares(global_chisquares)
        if include_utility:
            features = [(term, chisquares[term]) for term in chisquares]
        else:
            features = [term for term in chisquares]

        # Constrain max number of features.
        if max_features > 0 and len(features) > max_features:
            return features[:max_features]
        else:
            return features

    def get_local_features(self, category, threshold, include_utility=False, max_features=-1):
        chisquares = self.compute_chisquare_all(category)

        if include_utility:
            features = [(term, chisquares[term]) for term in chisquares if
                    chisquares[term] > threshold]
        else:
            features = [term for term in chisquares if chisquares[term] > threshold]

        # Constrain max number of features.
        if max_features > 0 and len(features) > max_features:
            return features[:max_features]
        else:
            return features

    def compute_chisquare_all(self, category, sort=True):
        chisquares = dict()
        terms = self.get_category_terms(category)
        for term in terms:
            chisquares[term] = self.compute_chisquare(term, category)

        if sort:
            return self.sort_chisquares(chisquares)
        else:
            return chisquares

    def compute_chisquare(self, term, category):
        """
        Chi-Square feature selection method.
        See: http://nlp.stanford.edu/IR-book/html/htmledition/feature-selectionchi2-feature-selection-1.html
        """

        if category not in self.training_data:
            raise UnknownCategoryException('Chi-squared value cannot be '
                'computed for unknown category "{0}"'.format(category))

        # n_11: Documents in the specified category and has the term.
        # n_10: Documents in the specified category and does not have the
        #       the term.
        n_11 = 0
        n_10 = 0
        category_data = self.training_data[category]
        for term_vector in category_data:
            if term in term_vector:
                n_11 += 1
            else:
                n_10 += 1

        # n_01: Documents not in the specified category and has the term.
        # n_00: Documents not in the specified category and does not have
        #       the term.
        n_01 = 0
        n_00 = 0
        for set_name in self.training_data:
            if category != set_name:
                set_data = self.training_data[set_name]
                for term_vector in set_data:
                    if term in term_vector:
                        n_01 += 1
                    else:
                        n_00 += 1
        
        # Find number of documents.
        doc_count = n_00 + n_01 + n_10 + n_11
        # print 'n: {0} | {1} | {2} | {3} ({4})'.format(n_11, n_10, n_01, n_00, term)

        """
        # Compute e_XX values.
        e_11 = doc_count * ((n_11 + n_10) / doc_count) * ((n_11 + n_01) / doc_count)
        e_10 = doc_count * ((n_10 + n_10) / doc_count) * ((n_10 + n_00) / doc_count)
        e_01 = doc_count * ((n_01 + n_00) / doc_count) * ((n_01 + n_10) / doc_count)
        e_00 = doc_count * ((n_00 + n_00) / doc_count) * ((n_00 + n_00) / doc_count)
        # print 'e: {0} | {1} | {2} | {3} ({4})'.format(e_11, e_10, e_01, e_00, term)
        
        # Compute chi-squared value.
        # Sometimes these values can be zero. In such a case, the chi-squared
        # value is actually infinity.
        if e_00 == 0 or e_01 == 0 or e_10 == 0 or e_11 == 0:
            chisquare = sys.float_info.max
        else:
            chisquare = ((n_00 - e_00) * (n_00 - e_00)) / e_00
            chisquare += ((n_01 - e_01) * (n_01 - e_01)) / e_01
            chisquare += ((n_10 - e_10) * (n_10 - e_10)) / e_10
            chisquare += ((n_11 - e_11) * (n_11 - e_11)) / e_11
        """

        chisquare = doc_count * ((n_11 * n_00) - (n_10 * n_01))
        chisquare /= (n_11 + n_10) * (n_01 + n_00) * (n_11 + n_01) * (n_10 + n_00)
        return chisquare

    def sort_chisquares(self, chisquares):
        return OrderedDict(sorted(chisquares.items(), key=lambda t: t[1],
            reverse=True))

class FrequencyBasedFeatureSelector(FeatureSelector):
    def __init__(self, training_data):
        self.global_threshold = 40
        super(FrequencyBasedFeatureSelector, self).__init__(training_data)

    def get_global_features(self, include_utility=False, max_features=-1):
        df = self.compute_global_df_all(True)

        if include_utility:
            features = [(term, df[term]) for term in df if df[term] > self.global_threshold]
        else:
            features = [term for term in df if df[term] > self.global_threshold]

        # Constrain max number of features.
        if max_features > 0 and len(features) > max_features:
            return features[:max_features]
        else:
            return features

    def compute_global_df(self, term):
        df = 0
        for category in self.training_data:
            category_data = self.training_data[category]
            for term_vector in category_data:
                if term in term_vector:
                    df += term_vector[term]
        
        return df

    def compute_global_df_all(self, sort=True):
        """
        We could have re-used compute_global_df, but then it will result in
        performance issues due to re-iteration over the entire training
        data set.
        """
        df = dict()
        for category in self.training_data:
            category_data = self.training_data[category]
            for term_vector in category_data:
                for term in term_vector:
                    if term not in df:
                        df[term] = term_vector[term]
                    else:
                        df[term] += term_vector[term]
       
        if sort:
            return self.sort_df(df)
        else:
            return df

    def get_local_features(self, category, threshold, include_utility=False,
            max_features=-1):
        df = self.compute_local_df_all(category)

        if include_utility:
            features = [(term, df[term]) for term in df if df[term] > threshold]
        else:
            features = [term for term in df if df[term] > threshold]

        # Constrain max number of features.
        if max_features > 0 and len(features) > max_features:
            return features[:max_features]
        else:
            return features

    def compute_local_df(self, term, category):
        """
        Computes the document frequency for a term in a category.
        This is the number of documents containing a particular term.
        """
        term = term.lower()
        df = 0

        # Check if category exists.
        if category not in self.training_data:
            raise UnknownCategoryException('Category "{0}" unknown'.format(
                category))

        # Compute the document frequency.
        for term_vector in self.training_data[category]:
            if term in term_vector:
                df += term_vector[term]
        
        return df

    def compute_local_df_all(self, category, sort=True):
        # Obtain all terms.
        df = dict()
        terms = self.get_category_terms(category)
        for term in terms:
            df[term] = self.compute_local_df(term, category)

        if sort:
            return self.sort_df(df)
        else:
            return df

    def sort_df(self, df):
        return OrderedDict(sorted(df.items(), key=lambda t: t[1],
            reverse=True))

# vim: set ts=4 sw=4 et:
