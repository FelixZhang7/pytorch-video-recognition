#!/usr/bin/env python
"""
Rand-related utility functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import sys
import math
import random
import numpy as np


def rand_idx(n, m, is_rand=True):
    """
    Randomly select m from 0 : n.

    Note:
      Slow if n is big. Use randInt instead.

    Input
      n    -  #total number
      m    -  #selected
      is_rand  -  randomly shuffle or not, {True} | False

    Output
      idx  -  index, m x
    """
    random.seed()
    idx = range(0, n)
    if is_rand:
      random.shuffle(idx)
    return idx[:m]


def rand_int(mi, ma, m):
    """
    Randomly select m integer from mi : ma.

    Input
      mi   -  minimum integer
      ma   -  maximum integer
      m    -  #selected

    Output
      res  -  result, m x
    """
    random.seed()
    res = []
    for i in range(m):
      a = random.randint(mi, ma)
      res.append(a)
    return res


def randp(n):
    """
    Get a randomly shuffle index array.

    Input
      n    -  #elements

    Output
      idx  -  index array, 1 x n
    """
    return rand_idx(n, n)
