#!/usr/bin/env python
"""
Position-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import numpy as np


def bin_to_idx(vals, bins):
    """
    Get the index for each bin.

    Input
      vals  -  list of values, n x
      bins  -  bin boundary, (k + 1) x

    Output
      idxs  -  index, k x
    """
    # dimension
    n = len(vals)
    k = len(bins) - 1

    idxs = [[] for i in xrange(k)]
    for i in xrange(n):

        opt_c = None
        for c in xrange(k):
            if c < k - 1:
                if vals[i] >= bins[c] and vals[i] < bins[c + 1]:
                    opt_c = c
                    break
            else:
                if vals[i] >= bins[c] and vals[i] <= bins[c + 1]:
                    opt_c = c
                    break

        if opt_c is None:
            print "warning, no bin found"
            continue

        idxs[opt_c].append(i)

    return idxs


def rangeG(n, m):
    """
    Get range groups.

    Example
      input:  n = 10, m = 3
      call:   rans = rangeG(n, m)
      output: rans = array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]], dtype=object)

    Input
      n     -  #total number
      m     -  #group size

    Output
      rans  -  nG x, m x
                 nG: #group = ceil(n / m)
    """
    # dimension
    st = range(0, n, m)
    st.append(n)
    nG = len(st) - 1

    # each group
    rans = np.empty(nG, dtype=object)
    for iG in range(nG):
      rans[iG] = range(st[iG], st[iG + 1])

    return rans
