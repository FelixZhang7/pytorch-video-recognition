#!/usr/bin/env python
"""
Classifier-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


def get_svm_feat(feats, labels, class_names):
    """
    Convert the feature to be used by SVM.

    Input
      feats        -  a list of feature, each element is of length d
      labels       -  a list of labels, each label is a string
      class_names  -  class names, k x

    Output
      X            -  feature matrix, n x d
      y            -  label vector, n x
    """
    # dimension
    n = len(feats)
    d = len(feats[0])
    k = len(class_names)

    # class_name dict
    dct = {class_name : id for id, class_name in enumerate(class_names)}

    X = np.zeros((n, d))
    y = np.zeros((n))
    for i in xrange(n):
        X[i] = feats[i]
        y[i] = dct[labels[i]]

    return X, y


def train_svm(X, y):
    """
    Train SVM.

    Input
      X    -  feature, n x d
      y    -  label, n x

    Output
      clf  -  classifier
    """
    clf = svm.LinearSVC(verbose=0)
    clf.fit(X, y)
    return clf


def train_svm_one(X, y, c):
    """
    Train SVM.
    """
    # adjust label
    n = len(y)
    for i in xrange(n):
        if y[i] == c:
            y[i] = 1
        else:
            y[i] = 0

    clf = svm.LinearSVC(verbose=0)
    clf.fit(X, y)

    return clf


def test_svm(clf, X):
    """
    Test SVM.

    Input
      clf  -  classifier
      X    -  feature, n x d

    Output
      Sco  -  Score, n x k
    """
    # compute score
    W = clf.coef_
    b = clf.intercept_

    # dimension
    n, _ = X.shape
    k = len(b)

    # predict
    Sco = np.dot(X, W.T) + np.tile(b, (n, 1))

    return Sco


def score_to_prediction(Sco):
    """
    Compute prediction from score.

    Input
      Sco  -  score, n x k

    Output
      cs   -  predicted label, n x
      C    -  predicted label sorted for all classes, n x k
    """
    # dimension
    n, k = Sco.shape

    # label
    cs = np.zeros((n), dtype=np.int)
    C = np.zeros((n, k), dtype=np.int)
    for i in xrange(n):
        C[i][...] = Sco[i].argsort()[-1 : -k - 1 : -1]
        cs[i] = C[i, 0]

    return cs, C


def confusion_matrix(cs, cTs, k):
    """
    Compute the confusion matrix.

    Input
      cs   -  prediction label, n x
      cTs  -  ground-truth label, n x
      k    -  #label

    Output
      D    -  confusion matrix, k x k
    """
    # dimension
    n = len(cs)

    # top-5 label for each instance
    co = 0
    vis = np.zeros((n))
    D = np.zeros((k, k), dtype=np.int)

    # each instance
    for i in xrange(n):
        # label for each instance
        c = int(cs[i])
        # C[i][...] = Sco[i].argsort()[-1 : -k - 1 : -1]

        # ground-truth label
        cT = int(cTs[i])

        # update confusion matrix
        D[cT, c] += 1

        # order
        # dsts = np.sum((ScoT - Sco) ** 2, axis=1)
        # vis1 = np.array(vis1) == 1
        # vis5 = np.logical_and(np.array(vis5) == 1, np.logical_not(vis1))
        # vis0 = np.logical_and(np.logical_not(vis1), np.logical_not(vis5))
        # idx0 = np.argsort(dsts[vis0])[::-1]
        # idx0 = np.nonzero(vis0)[0][idx0]
        # idx5 = np.argsort(dsts[vis5])[::-1]
        # idx5 = np.nonzero(vis5)[0][idx5]
        # idx1 = np.argsort(dsts[vis1])[::-1]
        # idx1 = np.nonzero(vis1)[0][idx1]

        # idx = np.concatenate((idx0, idx5, idx1))

    return D


def precision_recall(D):
    """
    Compute precision recall value from the confusion matrix.

    Input
      D     -  confusion matrix, k x k

    Output
      pres  -  precision value, k x
      recs  -  recall value, k x
    """
    k = D.shape[0]

    pres = np.zeros((k))
    recs = np.zeros((k))
    for c in xrange(k):
        true_pos = D[c, c]
        pres[c] = 1.0 * true_pos / max(D[:, c].sum(), 1)
        recs[c] = 1.0 * true_pos / max(D[c, :].sum(), 1)

    return pres, recs


def analyze_binary_scores(img_labels, pos_scores, pos_label):
    """
    Analyze binary scores.

    Input
      img_labels  -  image labels, num_img x
      pos_scores  -  positive score, num_img x
      pos_label   -  positive label

    Output
      stat        -  result
        pres      -  curve of precision
        recs      -  curve of recall
        ts        -  thresholds
        ap        -  average precision
    """
    import sklearn

    # get label vector
    num_img = len(img_labels)
    ys = []
    for i in xrange(num_img):
        if img_labels[i] == pos_label:
            ys.append(1)
        else:
            ys.append(0)

    # compute curve
    pres, recs, ts = sklearn.metrics.precision_recall_curve(ys, pos_scores)

    # compute ap
    ap = sklearn.metrics.average_precision_score(ys, pos_scores)

    # store
    from easydict import EasyDict as ED
    stat = ED()
    stat.pres = pres
    stat.recs = recs
    stat.ts = ts
    stat.ap = ap
    return stat
