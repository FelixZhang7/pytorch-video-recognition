#!/usr/bin/env python
"""
Other utility functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-03
"""
import sys
import time
import math
import random
from collections import OrderedDict
import numpy as np


def ismac():
  """
  Check whether it is on a Mac system.

  Output
    flag  -  True | False
  """
  return sys.platform == 'darwin'


def islinux():
  """
  Check whether it is on a Linux system.

  Output
    flag  -  True | False
  """
  return sys.platform.startswith('linux')


def quit():
  """
  Quit if is not mac.
  """
  if not ismac():
    sys.exit()


def tic():
  """
  Mimick Matlab's tic function
  """
  ti = time.time()
  return ti


def toc(ti0):
  """
  Mimick Matlab's toc function

  Input
    ti0  -  starting time

  Output
    ti   -  duration
  """
  ti = time.time()
  return ti - ti0


def vec2img(vec):
  """
  Convert a vector to a squared (h == w) img matrix.
  Used for cuda convnet format.

  Input
    vec  -  vector, d x 1

  Output
    Img  -  image matrix, h x w x 3 (uint8)
  """
  pixels = vec.shape[0] / 3
  size = int(math.sqrt(pixels))
  Img = vec.reshape((3, size, size)).swapaxes(0, 2).swapaxes(0, 1)
  return Img


def fact(n):
  """
  Returns all the prime factors of a positive integer.

  Input
    n        -  integer

  Output
    factors  -  list of prime factors
  """
  factors = []
  d = 2
  while n > 1:
      while n % d == 0:
          factors.append(d)
          n /= d
      d = d + 1

  return factors


def equal(nm, A, B):
  """
  Check whether two matrix are equal or not.

  Input
    nm    -  name
    A     -  matrix 1
    B     -  matrix 2

  Output
    isEq  -  result, True | False
    msg   -  message
  """
  # check type
  if not A.__class__ == B.__class__:
    msg = '{}, different types: {} vs {}'.format(nm, A.__class__, B.__class__)
    return False, msg

  # dictionary, check each key
  if A.__class__ == dict:
    keyAs = A.keys()
    keyBs = A.keys()
    keyAs.sort()
    keyBs.sort()
    if not equal(nm, keyAs, keyBs):
      return False, None
    for key in keyAs:
      if not equal(nm, A[key], B[key]):
        return False, None
    return True, None

  # list
  if A.__class__ == list:
    if not len(A) == len(B):
      msg = '{}, different lens: {} vs {}'.format(nm, len(A), len(B))
      return False, msg

    for i in range(len(A)):
      if not equal(nm, A[i], B[i]):
        return False, None
    return True, None

  # string
  if A.__class__ == str:
    if not A == B:
      msg = '{}, different strings: {} vs {}'.format(nm, A, B)
      return False, msg
    return True, None

  # check dimension
  if not A.shape == B.shape:
    msg = '{}, different size: {} vs {}'.format(nm, A.shape, B.shape)
    return False, msg

  # check value
  diff = max(abs((A - B).flatten()))
  isEq = diff < 1e-5
  if isEq:
    msg = '{}, same'.format(nm)
  else:
    msg = '{}, different values: diff = {}'.format(nm, diff)

  return isEq, msg


def listElems(list0, idx):
  """
  Return a sub list corresponding to the index.

  Input
    list0  -  original list, 1 x n
    idx    -  index, 1 x m

  Output
    list   -  new list, 1 x m
  """

  list = [list0[id] for id in idx]
  return list


def dictMerge(dict1, dict2):
  """
  Merge two dictionaries.

  Input
    dict1  -  1st dictionary
    dict2  -  2nd dictionary

  Output
    dict   -  output dictionary
  """
  dict = {}
  for key in dict1.keys():
    dict[key] = dict1[key] + dict2[key]

  return dict


def randIdx(n, m, isR=True):
  """
  Randomly select m from 0 : n.

  Note:
    Slow if n is big. Use randInt instead.

  Input
    n    -  #total number
    m    -  #selected
    isR  -  randomly shuffle or not, {True} | False

  Output
    idx  -  index, m x
  """
  random.seed()
  idx = range(0, n)
  if isR:
    random.shuffle(idx)
  return idx[:m]


def randInt(mi, ma, m):
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
  return randIdx(n, n)


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


def splitLst2(lst, rat, isR=True):
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


def lstUni(lst0):
  """
  Return a unique list.

  Input
    lst0  -  original list, n0 x 1

  Output
    lst   -  new list, n x 1
    c0s   -  new label of the original list, n0 x 1
    idx   -  index of the new list in the original list, n x 1
    dct   -  {label name : label id}
  """
  lst = []
  idx = []
  dct = {}
  c0s = []
  k = 0;
  for i, item in enumerate(lst0):
    if dct.has_key(item):
      c0s.append(dct[item])
      continue

    dct[item] = k
    lst.append(item)
    idx.append(i)
    c0s.append(k)
    k += 1

  return lst, c0s, idx, dct


def mapSort(dct0, alg):
  """
  Return the key list sorted according to the specified method.

  Input
    dct0  -  original dictionary
    alg   -  sorted algorithm, {'key'} | 'valS2L' | 'valL2S' |
               'key': key
               'valS2L': value from small to large
               'valL2S': value from large to small

  Output
    dct   -  sorted dictionary (OrderedDict)
  """
  # original key and value list
  key0s = dct0.keys()
  val0s = dct0.values()

  if alg.startswith('key'):
    if alg.endswith('S2L'):
      idx = np.argsort(key0s)
    else:
      idx = np.argsort(key0s)[::-1]

  elif alg.startswith('val'):
    if alg.endswith('S2L'):
      idx = np.argsort(val0s)
    else:
      idx = np.argsort(val0s)[::-1]

  else:
    raise Exception('unknown alg: {}'.format(alg))

  dct = OrderedDict()
  for id in idx:
    dct[key0s[id]] = val0s[id]
  return dct


def lstNm2Id(lst0, nms):
  """
  Convert the values of list to id.

  Input
    lst0  -  original list, m
    nms   -  value list, k

  Output
    lst   -  new list, m
  """
  # build a dictionary for lst
  nm2id = {nm : id for id, nm in enumerate(nms)}

  lst = [nm2id[nm] for nm in lst0]

  return lst


def mapCo(dct):
  """
  Count the frequency of a string map.

  Input
    dct    -  input dict

  Output
    coMap  -  a dict for storing counts
  """
  # all the values
  vals = []
  for vali in dct.values():
    vals += vali

  # init
  coMap = {val : 0 for val in vals}

  # count
  for key in dct.keys():
    vali = dct[key]
    for val in vali:
      coMap[val] += 1

  return coMap


def mapValNm2Id(dct0, nms):
  """
  Convert the values of map to id.

  Input
    dct0  -  original map, m : mi
    nms   -  value list, k

  Output
    dct   -  new map, m : mi
  """
  # build a dictionary for nms
  nm2id = {nm : id for id, nm in enumerate(nms)}

  dct = {}
  for key in dct0.keys():
    val = dct0[key]
    dct[key] = nm2id[val]

  return dct


def mapValLstNm2Id(dct0, nms):
  """
  Convert the values of map to id.

  Input
    dct0  -  original map, m : mi
    nms   -  value list, k

  Output
    dct   -  new map, m : mi
  """
  # build a dictionary for nms
  nm2id = {nm : id for id, nm in enumerate(nms)}

  dct = {}
  for key in dct0.keys():
    vals = dct0[key]
    dct[key] = [nm2id[val] for val in vals]

  return dct


def objBaseTree(obj):
  """
  Obtain the base-class tree of an object.

  Input
    obj   -  object

  Output
    tree  -  class tree
  """
  import inspect
  tree = inspect.getclasstree(inspect.getmro(obj.__class__))

  return tree


def appendConst2Array(a0, d, v=1):
  """
  Append a const value to an array.

  Input
    a0  -  original array, d0 x
    d   -  new dimension
    v   -  value, {1} | ...

  Output
    a   -  new array, d x
  """
  d0 = len(a0)
  if d0 > d:
    raise Exception('unknown case')

  a = []
  for i in range(d):
    if i < d0:
      a.append(a0[i])
    else:
      a.append(v)

  return a


def maxGeneral(a, ma=0):
  """
  A general max function.
  It can handle the case when input is empty.

  Input
    a   -  array, n x
    ma  -  default maximum value, {0} | ...
  """
  if a is None or len(a) == 0:
    return ma
  return max(a)


def minGeneral(a, mi=0):
  """
  A general min function.
  It can handle the case when input is empty.

  Input
    a   -  array, n x
    mi  -  default minimum value, {0} | ...
  """
  if a is None or len(a) == 0:
    return mi
  return min(a)
