#!/usr/bin/env python
"""
Dictionary-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
from easydict import EasyDict as ED


def vals(dcts, key):
    """
    Given a list of dictionary, return a list of values corresponding to the key.

    Input
      dcts  -  list of dictionary, n x
      key   -  key string

    Output
      vals  -  value list, n x
    """
    return [dct[key] for dct in dcts]


def sub(dct0, keys):
    """
    Return a sub-dictionary for the selected keys.

    Input
      dct0  -  original dictionary
      keys  -  key list, m x

    Output
      dct   -  new dictionary
    """
    return dict((key, dct0[key]) for key in keys)


def dctItem(dct, idx):
    """
    Return a list of items for the corresponding index.

    Input
      dct   -  original dictionary
      idx   -  index, m x

    Output
      keys  -  keys, m x
      vals  -  values, m x
    """
    keys = []
    vals = []
    items = dct.items()
    for i in idx:
        key, val = items[i]
        keys.append(key)
        vals.append(val)
    return keys, vals


def lns2dct(lns, sep=':'):
    """
    Generate dictionary from lines.

    Input
      lns  -  line, n x
      sep  -  seperator, {':'} | ...

    Output
      dct  -  dictionary
    """
    dct = {}
    for ln in lns:
        parts = ln.split(sep)
        assert(len(parts) == 2)
        dct[parts[0]] = parts[1]

    return dct


def ps(dct, key, val0):
    """
    Parse the parameter specified in a struct or in a cell array.

    Example 1 (when option is a struct):
      input    -  dct['lastname'] = 'zhou';
      call     -  value = ps(dct, 'lastname', 'noname');
      output   -  value = 'zhou'

    Example 2 (when option is a cell array):
      input    -  option = {'lastname', 'zhou'};
      call     -  value = ps(option, 'lastname', 'noname');
      output   -  value = 'zhou'

    Input
      dct      -  dictionary
      key      -  filed name
      val0     -  default field value
    """
    if dct is None:
        return val0

    if not dct.has_key(key):
        return val0

    return dct[key]
