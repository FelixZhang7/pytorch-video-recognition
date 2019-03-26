#!/usr/bin/env python
"""
Matrix-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import numpy as np


def add_up_row(A, row_ids_remove, row_id_target):
    """
    Add up rows into one row.

    The result matrix will have the following size
      num_rows = num_rows0 - num_idx + 1

    Input
      A0       -  matrix, num_rows0 x num_cols
      row_ids_remove  -  index, num_idx
      row_id_target  -  index, num_idx

    Output
      A        -  new matrix, num_rows x num_cols
    """
    A_sub = A[row_ids_remove, :].sum(axis=0)
    A[row_id_target] += A_sub
    B = np.delete(A, row_ids_remove, axis=0)

    return B


def calc_dst(X1, X2, dst='e'):
    """
    Compute distance matrix.

    Remark
      Dij is the squared Euclidean distance between the i-th point in X1 and j-th point in X2.
      i.e., D[i, j] = || X1[i] - X2[j] ||_2^2

    Usage
      input  -  X1 = np.random.rand(5, 3)
                X2 = np.random.rand(6, 3)
      call   -  D = calc_dist(X1, X2)
                D.shape = (5, 6)

    Input
      X1     -  1st sample matrix, n1 x dim | dim x
      X2     -  2nd sample matrix, n2 x dim | dim x
      dst    -  distance type, {'e'} | 'b'
                  'e': Euclidean distance
                  'b': binary distance
    Output
      D      -  squared distance matrix, n1 x n2
    """
    # dimension
    n1, dim = X1.shape
    n2 = X2.shape[0]

    if dim == 1:
        X1 = np.concatenate((X1, np.zeros((n1, 1))), axis=1)
        X2 = np.concatenate((X2, np.zeros((n2, 1))), axis=1)

    XX1 = np.expand_dims((X1 * X1).sum(axis=1), axis=1)
    XX2 = np.expand_dims((X2 * X2).sum(axis=1), axis=0)

    # compute
    X12 = np.dot(X1, X2.T)
    D = np.tile(XX1, (1, n2)) + np.tile(XX2, (n1, 1)) - 2 * X12

    # Euclidean distance
    if dst == 'e':
        pass
    else:
        raise Exception('unknown distance: {}'.format(dst));

    return D


def calc_bandwidth(D, nei=0.2):
    """Compute the bandwidth for RBF kernel.

    Input
      D      -  squared distance matrix, n x n
      nei    -  #nearest neighbour to compute the kernel bandwidth, {.1}

    Output
      sigma  -  kernel bandwidth (variance)
    """
    # dimension
    n = D.shape[0]
    m = min(max(1, int(np.floor(nei * n))), n)

    # nearest neighbors
    Dsorted = np.sort(D, axis=0)

    D2 = np.real(np.sqrt(Dsorted[0 : m, :]))
    sigma = D2.sum() / (n * m)

    return sigma


def calc_kernel(D, nei=0.2):
    """Construct the RBF kernel matrix from squared distance matrix.

    Input
      D    -  squared distance matrix, n x n
      nei  -  #nearest neighbour to compute the kernel bandwidth, {.1}
                see function "calc_bandwidth" for more details
    Output
      K    -  kernel matrix, n x n
    """
    # bandwidth
    sigma = calc_bandwidth(D, nei=nei)

    # kernel
    K = np.exp(-D / (2 * sigma * sigma + np.finfo(float).eps))

    return K
