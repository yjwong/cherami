#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import logging
import tempfile
import shutil

import nltk
from nltk.corpus import stopwords
from nltk.downloader import Downloader

import config
from exception import UnknownStopwordSourceException

logger = logging.getLogger(__name__)

class TweetTokenizer:
    def tokenize(self, tweet):
        raise NotImplementedError()

class SimpleTokenizer(TweetTokenizer):
    def tokenize(self, tweet):
        return tweet.split()

class StopwordRemover:
    def __init__(self):
        self.stopword_list = set()

    def remove(self, terms):
        return [i for i in terms if i not in self.stopword_list]

    def remove_mentions(self, terms):
        return [i for i in terms if not i.startswith('@')]

    def build_lists(self):
        sources = config.stopword_sources
        for source in sources:
            source_type, source_attrib = source.split(':')
            if source_type == 'nltk':
                logger.info('Building stopword list from NLTK using corpus '
                    '"{0}"...'.format(source_attrib))
                self.build_list_from_nltk(source_attrib)

            elif source_type == 'file':
                f = open(source_attrib, 'r')
                for line in f:
                    self.stopword_list.add(line.strip())
            
            else:
                raise UnknownStopwordSourceException('Unknown stopword source '
                        'type "' + source_type + '".')

        logger.info('{0} stopwords added. '.format(len(self.stopword_list)))

    def build_list_from_nltk(self, lang):
        downloader = Downloader()
        tempdir = None
        
        # Check if the NLTK data has already been downloaded.
        if not downloader.is_installed('stopwords'):
            # Create temporary directory for download
            tempdir = tempfile.mkdtemp(prefix='cherami')
            logger.info('Downloading NLTK stopword data into "{0}"'
                '...'.format(tempdir))

            downloader.download('stopwords', tempdir, True)
            logger.info('NLTK stopword data downloaded.')

            nltk.data.path = [tempdir]

        for word in stopwords.words(lang):
            self.stopword_list.add(word)

        # Clean up after we're done.
        if tempdir is not None:
            shutil.rmtree(tempdir)

# vim: set ts=4 sw=4 et:
