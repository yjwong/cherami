#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import re
import logging
import tempfile
import shutil
import HTMLParser

import nltk
from nltk.corpus import stopwords
from nltk.downloader import Downloader

import config
from exception import UnknownStopwordSourceException

logger = logging.getLogger(__name__)

class TweetTokenizer:
    """An interface for a tokenizer for a tweet."""

    def tokenize(self, tweet):
        raise NotImplementedError()

class SimpleTokenizer(TweetTokenizer):
    """A simple string-based tokenizer that implements TweetTokenizer.""" 

    def tokenize(self, tweet):
        return tweet.split()

class NLTKTokenizer(TweetTokenizer):
    """A tokenizer based on NLTK's wordpunct_tokenize module."""

    def tokenize(self, tweet):
        return nltk.wordpunct_tokenize(tweet)

class TweetTextFilter:
    """
    A text filter for tweets.

    Removes redundant information such as mentions, links and symbols from the
    tweet body. In addition, HTML entities are converted into their normal
    forms and the entire tweet body is converted to lowercase.
    """

    def filter(self, text):
        text = self.convert_entities(text)
        text = self.remove_mentions(text)
        text = self.remove_links(text)
        text = self.remove_symbols(text)
        text = self.convert_case(text)
        return text

    def remove_mentions(self, text):
        return re.sub('@([A-Za-z]+[A-Za-z0-9_]*)', '', text)

    def remove_links(self, text):
        return re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    def remove_symbols(self, text):
        return re.sub(r'([^A-Za-z0-9\'])', ' ', text)

    def convert_entities(self, text):
        parser = HTMLParser.HTMLParser()
        return parser.unescape(text)

    def convert_case(self, text):
        return text.lower()

class StopwordRemover:
    """
    A stopword remover.

    Removes words that are extremely common and/or do not provide strong
    significance to the meaning/topic suggested by the tweet. The old-style
    RTs are removed, along with words appearing in NLTK's stopword list.
    Tokens containing only numbers are also removed.

    TODO: Language detection.
    """

    # Keep track of references to this class so that we know when to remove
    # the NLTK data.
    refcnt = 0
    nltk_dir = None

    def __init__(self):
        StopwordRemover.refcnt += 1
        logger.debug('refcnt is {0}'.format(StopwordRemover.refcnt))

        self.stopword_list = set()

    def __del__(self):
        StopwordRemover.refcnt -= 1
        logger.debug('refcnt is {0}'.format(StopwordRemover.refcnt))

        if StopwordRemover.nltk_dir != None and StopwordRemover.refcnt is 0:
            logger.info('No instance of StopwordRemover remains, removing '
                'NLTK data directory')

            shutil.rmtree(StopwordRemover.nltk_dir)
            StopwordRemover.nltk_dir = None

    def remove_all(self, terms, tweet=None):
        terms = self.remove_rt(terms)
        terms = self.remove(terms)
        terms = self.remove_numbers(terms)
        terms = self.remove_short(terms)
        return terms

    def remove(self, terms):
        return [i for i in terms if i not in self.stopword_list]

    def remove_rt(self, terms):
        return [i for i in terms if not i == 'rt']

    def remove_short(self, terms):
        return [i for i in terms if len(i) >= 3]

    def remove_numbers(self, terms):
        filtered_terms = list()
        for i in terms:
            try:
                float(i)
            except ValueError:
                filtered_terms.append(i)
                pass

        return filtered_terms

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

        # Check if NLTK data directory exists.
        if StopwordRemover.nltk_dir == None:
            # Create temporary directory for download
            StopwordRemover.nltk_dir = tempfile.mkdtemp(prefix='cherami')
            nltk.data.path = [StopwordRemover.nltk_dir]
            
            logger.info('NLTK data directory is "{0}"'
                .format(StopwordRemover.nltk_dir))
        
        # Check if the NLTK data has already been downloaded.
        if not downloader.is_installed('stopwords'):
            logger.info('Downloading NLTK stopword data...')
            downloader.download('stopwords', StopwordRemover.nltk_dir, True)
            logger.info('NLTK stopword data downloaded.')

        for word in stopwords.words(lang):
            self.stopword_list.add(word)

class VocabNormalizer:
    def __init__(self):
        self.vocab_map = dict()

    def normalize(self, terms):
        for i in range(len(terms)):
            if terms[i] in self.vocab_map:
                terms[i] = self.vocab_map[terms[i]]

        return terms

    def build_map(self):
        source = config.vocab_map_file
        f = open(source, 'r')
        for line in f:
            key, value = line.split()
            key = key.strip()
            value = value.strip()
            self.vocab_map[key] = value

# vim: set ts=4 sw=4 et:
