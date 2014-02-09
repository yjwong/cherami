#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

from setuptools import setup

setup(name='cherami',
      version='0.1',
      description='A Tweet Classifier',
      long_description='Cher Ami is a project that analyses and classifies '
        'tweets from Twitter. This project is an assignment for CS4242 Social '
        'Media Computing, National University of Singapore.',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Education',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      url='https://github.com/yjwong/cherami',
      author='Wong Yong Jie',
      author_email='yjwong92@gmail.com',
      license='Apache',
      packages=['cherami'],
      install_requires=[
          'tweepy'
      ],
      zip_safe=False)

# vim: set ts=4 sw=4 et:
