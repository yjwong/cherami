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
    pass

class CommandLineException(CheramiException):
    pass

class UnknownStopwordSourceException(CheramiException):
    pass

class ClassifierNotTrainedException(CheramiException):
    pass

# vim: set ts=4 sw=4 et:
