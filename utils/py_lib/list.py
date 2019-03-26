#!/usr/bin/env python
"""
Some utility functions for list operation.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import random


def split2(lst, rat, isR=True):
    """
    Split a list into two parts.
    The size of the 1st set is rat * len(lns), and the 2nd is (1 - rat) * len(lns)

    Input
      lst   -  original list, 1 x n
      rat   -  ratio, float < 1 | int > 1
      isR   -  randomly shuffle or not, {True} | False

    Output
      lst1  -  1st set, 1 x n1
      lst2  -  2nd set, 1 x n2
    """
    # dimension
    n = len(lst)

    # index
    idx = range(n)
    if isR:
      random.shuffle(idx)

    # split index
    if rat < 1:
      n1 = int(rat * n)
    else:
      n1 = int(rat)
    lst1 = [lst[i] for i in idx[: n1]]
    lst2 = [lst[i] for i in idx[n1 :]]

    return lst1, lst2


def concate_in_lines(lists, sep=' '):
    """
    Split a list into two parts.
    The size of the 1st set is rat * len(lns), and the 2nd is (1 - rat) * len(lns)

    Input
      lst   -  original list, 1 x n
      rat   -  ratio, float < 1 | int > 1
      isR   -  randomly shuffle or not, {True} | False

    Output
      lst1  -  1st set, 1 x n1
      lst2  -  2nd set, 1 x n2
    """
    # dimension
    num_list = len(lists)
    num_line = len(lists[0])

    lines = []
    for line_id in xrange(num_line):
        line = ''
        for list_id in xrange(num_list):
            if list_id > 0:
                line += sep
            line += '{}'.format(lists[list_id][line_id])
        lines.append(line)

    return lines


def split_from_lines(lines, sep=' '):
    """
    Split a list into two parts.
    The size of the 1st set is rat * len(lns), and the 2nd is (1 - rat) * len(lns)

    Input
      lst   -  original list, 1 x n
      rat   -  ratio, float < 1 | int > 1
      isR   -  randomly shuffle or not, {True} | False

    Output
      lists
    """
    # dimension
    num_line = len(lines)

    lists = []
    for line_id in xrange(num_line):
        items = lines[line_id].split(sep)
        if line_id == 0:
            num_list = len(items)
            for list_id in xrange(num_list):
                lists.append([])
        else:
            assert num_list == len(items)

        for list_id in xrange(num_list):
            lists[list_id].append(items[list_id])

    return lists


def argsort(seq):
    """Argsort for list.

    http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python

    Input
      seq  -  list

    Output
      idx  -  index
    """
    return sorted(range(len(seq)), key=seq.__getitem__)
