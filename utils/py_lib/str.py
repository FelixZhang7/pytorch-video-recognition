#!/usr/bin/env python
"""
Utility functions for string operation.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import re
import os
import os.path as osp


def endswith_any(str, subxs, case_in=True):
    """
    Check whether the given string 'str' is ended with any subfix specified in 'subxs'.

    Input
      str      -  string
      subxs    -  subfix array, {None} | [] | 'txt' | 'bmp' | ...
      case_in  -  match in a case-insensitive way, {True} | False

    Output
      is_end   -  result, True | False
    """
    if subxs is None:
        return False

    if case_in:
        str = str.lower()

    for subx in subxs:
        if case_in:
            subx = subx.lower()

        # check
        if str.endswith(subx):
            return True

    return False


def lst_pat(lst0, pats):
    """
    Return a sub-list of string that match with the specified pattern.

    Input
      lst0  -  original string list, n0 x
      pats  -  pattern list, m x

    Output
      lst   -  new string list, n x
    """
    lst = []
    for str0 in lst0:
        for pat in pats:
            if re.match(pat, str0):
                lst.append(str0)
                break

    return lst


def get_subfix(name0):
    """Get subfix from a file name.

    Input
      name0  -  original name

    Output
      name   -  new name
    """
    tail = name0.rfind('.')
    if tail == -1:
        return None
    else:
        return name0[tail + 1:]


def del_subfix(name0):
    """
    Remove subfix from a file name.

    Input
      name0  -  original name

    Output
      name   -  new name
    """
    tail = name0.rfind('.')
    if tail == -1:
      name = name0
    else:
      name = name0[: tail]

    return name


def replace_subfix(name0, subx):
    """
    Replace subfix to the given one.

    Input
      name0  -  original name
      subx   -  new subx

    Output
      name   -  new name
    """
    name = del_subfix(name0)

    return name + '.' + subx


def append_subfix_before_extension(name0, subx):
    """
    Append string before extension.

    Input
      name0  -  original name
      subx   -  subfix

    Output
      name   -  new name
    """
    ext = get_subfix(name0)
    name = del_subfix(name0)

    return name + subx + '.' + ext


def str2ran(s):
    """Convert a string range to an integer list.

    Example 1
      input: s = '1'
      call:  lst = str2ran(s)
      output: lst = 1

    Example 2
      input: s = '2:10'
      call:  lst = str2ran(s)
      output: lst = [2, 3, 4, 5, 6, 7, 8, 9]

    Example 3
      input: s = '2:10:2'
      call:  lst = str2ran(s)
      output: lst = [2, 4, 6, 8]

    Example 4
      input: s = '1,3'
      call:  lst = str2ran(s)
      output: lst = [1,3]

    Example 5
      input: s = ''
      call:  lst = str2ran(s)
      output: lst = []

    Input
      s    -  string

    Output
      lst  -  an integer list
    """
    if len(s) == 0:
        lst = []

    elif ':' in s:
        parts = s.split(':')
        a = [int(part) for part in parts]

        if len(parts) == 1:
            lst = a
        elif len(parts) == 2:
            lst = range(a[0], a[1])
        elif len(parts) == 3:
            lst = range(a[0], a[1], a[2])
        else:
            raise Exception('unsupported')

    elif ',' in s:
        parts = s.split(',')
        lst = [int(part) for part in parts]

    else:
        lst = [int(s)]

    return lst
