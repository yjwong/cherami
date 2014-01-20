#!/usr/bin/env python

# CS4242 Social Media Computing
# Assignment 1
# See LICENSE for details

import iso8601

def ISO8601toTwitterDate(iso8601_date):
    parsed_date = iso8601.parse_date(iso8601_date)
    return parsed_date.strftime("%a %b %d %H:%M:%S %z %Y")

# vim: set ts=4 sw=4 et:
