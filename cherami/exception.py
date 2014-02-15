#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

class CheramiException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ConfigException(CheramiException):
    """ Raised when there is a configuration error. """
    pass

class CommandLineException(CheramiException):
    """ Raised when invalid parameters are specified on the command line. """
    pass

class UnknownStopwordSourceException(CheramiException):
    """ Raised when the stopword source is not a known source. """
    pass

class ClassifierNotTrainedException(CheramiException):
    """
    Raised when the classifier is not trained and the called method requires a
    trained classifier.
    """
    pass

class UnknownCategoryException(CheramiException):
    """ Raised when the classification category is not known. """
    pass

# vim: set ts=4 sw=4 et:
