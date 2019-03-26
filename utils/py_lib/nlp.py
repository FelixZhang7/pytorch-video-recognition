#!/usr/bin/env python
"""
NLP-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-11
"""
import numpy as np

def count_same_char(a, b, ignore_chars=[], allow_dup=True):
    """Counting the number of same chars shared in both strings.

    Input
      a             -  string a
      b             -  string b
      ignore_chars  -  chars to ignore, {[]}
      allow_dup     -  flag of allowing duplicate matching

    Output
      count         -  count
    """
    a = a.decode('utf-8')
    b = b.decode('utf-8')
    len_b = len(b)

    # record char that has been matched before
    if not allow_dup:
        vis = np.zeros((len_b))

    count = 0

    # loop over all chars in a
    for i in xrange(len(a)):
        # skip ignore chars
        if a[i] in ignore_chars:
            continue

        # loop over all chars in b
        for j in xrange(len(b)):
            # skip ignore chars
            if b[j] in ignore_chars:
                continue

            # found same
            if a[i] == b[j]:
                if not allow_dup and vis[j] == 1:
                    continue

                count += 1

                if not allow_dup:
                    vis[j] = 1
                break
    return count
