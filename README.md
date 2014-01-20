Cher Ami: A Tweet Classifier
======

Cher Ami is a project that analyses and classifies tweets from Twitter. This
project is an assignment for CS4242 Social Media Computing, National University
of Singapore.

The project is named after
[a homing pigeon that saved many lives](http://en.wikipedia.org/wiki/Cher_Ami)
during World War I in 1918.

Installation
------------
Installation is not required. However, the following dependencies are required
to run the classifier:

 * [tweepy](https://github.com/tweepy/tweepy), a library for Twitter
 * [iso8601](https://pypi.python.org/pypi/iso8601/), a library to parse ISO 8601
   formatted dates

Configuration
-------------
The training set is provided as a file.

There are sources where tweets can be obtained:

 * Link (A status link is supplied)
 * Twitter (Uses Twitter's Streaming API)
 * File (Uses JSON data from a file)

To use Twitter's Streaming API, one needs to be registered on
[Twitter Developers](https://dev.twitter.com/) in order to obtain the required
information:

 * OAuth consumer key
 * OAuth consumer secret
 * OAuth access token key
 * OAuth access token secret

Modify `cherami/config.py` accordingly.

Running
-------
The classifier can be run by using the following command:

    python cherami/cherami.py

Documentation
-------------
There is no documentation provided at the moment.

