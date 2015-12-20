#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import heapq
import argparse
import operator
import numpy as np
from math import sqrt
from scipy.linalg import svd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from utilities import *


VERSION = 'v4.0.0'
USAGE = '''usage: %(prog)s [options] arg1 arg2'''


def get_args():
    """TODO: Docstring for get_args.
    :returns: TODO

    """
    parser = argparse.ArgumentParser(usage=USAGE)

    parser.add_argument('--version', action='version',
                        help='Version.', version=VERSION)
    parser.add_argument('-d', action='store', type=int, default=None,
                        help='Preserved Dimention.', dest='d')
    parser.add_argument('-v', action='store', type=float, default=None,
                        help='Preserved Variance.', dest='v')
    parser.add_argument('--cfk', action='store', type=int, default=10,
                        help='K nearest neighbors.', dest='cfk')
    parser.add_argument('--fn', action='store',
                        help='The name of the file.', dest='fn')
    parser.add_argument('--movies', action='store',
                        help='The name of the movies file.', dest='moviesf')
    parser.add_argument('--rating', action='store',
                        help='The name of the rating file.', dest='ratingf')
    parser.add_argument('--users', action='store',
                        help='The name of the users file.', dest='usersf')
    parser.add_argument('--forhomework', action='store_true',
                        help='Homework model.', dest='forhomework')

    return parser.parse_args()


def cal_sim(a, b, ref):
    """TODO: Docstring for cal_sim.
    :returns: TODO

    """
    if b[ref] == 0:
        return 0

    a = a / np.sum(a[a!=0], dtype=np.float)
    b = b / np.sum(b[b!=0], dtype=np.float)
    res = pearsonr(a, b)[0]

    # a or b is zeros.
    if np.isnan(res):
        return 0

    return res


def cal_rmse(y_actual, y_pred):
    """TODO: Docstring for cal_rmse.
    :returns: TODO

    """
    rmse = sqrt(mean_squared_error(y_actual, y_pred))
    return rmse


def get_sims(idx, ref, X, k, item2item=True):
    """TODO: Docstring for get_sims.
    :returns: TODO

    """
    # TODO itself
    if item2item:
        x = X.T
        item = X[:,idx]
    else:
        x = X
        item = X[idx,:]
    neighbors = zip(*heapq.nlargest(k, enumerate([cal_sim(item,a, ref)
                                                  for a in x]),
                                    key=operator.itemgetter(1)))
    return neighbors[0], neighbors[1]


def rateit(first, second, X, k, item2item=True):
    """TODO: Docstring for rateit.
    :returns: TODO

    """
    indices, sims = get_sims(second, first, X, k, item2item)
    user_mean = np.mean(X[first,X[first,:]!=0], dtype=np.float)
    bxi = user_mean + np.std(X[:,second])
    return bxi + np.sum(sims*((X[first, indices]-user_mean)-\
                              np.std(X[:,indices],axis=0)))/np.sum(sims)


def baseline_estimator(X):
    """TODO: Docstring for baseline_estimator.
    :returns: TODO

    """
    for idx, rate in enumerate(X[0,:]):
        if rate != 0:
            print rate, rateit(0, idx, X, 20)
    # print rateit(0,0,X,20)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = get_args()
    if args.forhomework:
        X_train, X_test = transform_data(args.moviesf, args.ratingf,
                                         args.usersf, True)
    else:
        X = load_data(args.fn, 'rating').values

    print X_train.shape, X_test.shape

    start = time.time()
    baseline_estimator(X_train)
    print 'recsys time: %f' % (time.time() - start)


if __name__ == "__main__":
    main()
