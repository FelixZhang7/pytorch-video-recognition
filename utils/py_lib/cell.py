#!/usr/bin/env python
"""
Cell-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import numpy as np


def cells(shape, n=1):
    """
    Create a cell matrix.

    Example
      a, b, c = cells((3), n=3)

    Input
      shape  -  shape tuple
      n      -  #output, {1} | 2 | ...

    Output
      res    -  cell matrix, n x
    """
    if n == 0:
        return

    res = tuple(np.empty(shape, dtype=np.object) for i in range(n))

    if n == 1:
        return res[0]

    else:
        return res


def lists(len, n=1):
    """
    Create a list of arrays.

    Example
      a, b = lists(3, n=2)
      a = [[], [], []]
      b = [[], [], []]

    Input
      len  -  shape tuple
      n    -  #output, {1} | 2 | ...

    Output
      res  -  list, n x
    """
    if n == 0:
        return

    res = []
    for i in range(n):
        resi = []
        for j in range(len):
            resi.append([])
        res.append(resi)

    if n == 1:
        return res[0]

    else:
        return res


def zeros(shape, n=1):
    """
    Create n zero numpy matrices.

    Example
      a, b, c = zeros((3, 4), n=3)

    Input
      shape  -  shape tuple
      n      -  #output, {1} | 2 | ...

    Output
      res    -  zero matrix, n x
    """
    if n == 0:
        return

    res = tuple(np.zeros(shape) for i in range(n))

    if n == 1:
        return res[0]

    else:
        return res


def rands(shape, n=1):
    """
    Create random numpy matrices.

    Input
      shape  -  shape tuple
      n      -  #output, {1} | 2 | ...

    Output
      A      -  rand matrix
    """
    if n == 0:
        return

    res = tuple(np.random.rand(*shape) for i in range(n))

    if n == 1:
        return res[0]
    else:
        return res


def ods(n=1):
    """
    Create n OrderedDict objects.

    Input
      n      -  #output, {1} | 2 | ...

    Output
      o      -  OrderedDict
    """
    if n == 0:
        return

    import collections as col
    res = tuple(col.OrderedDict() for i in range(n))

    if n == 1:
        return res[0]
    else:
        return res
