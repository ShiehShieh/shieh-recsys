#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import math
import argparse
import numpy as np
from scipy.linalg import svd
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


def dim_reduction_svd(X, d=None, v=None):
    """TODO: Docstring for dim_reduction_svd.
    :returns: TODO

    """
    U, Sigma, Vh = svd(X)

    if v:
        sum_ = 0
        summation = np.sum(Sigma)
        for i in range(Sigma.shape[0]):
            sum_ += Sigma[i]
            if sum_/summation >= v:
                idx = i
                break
    elif d:
        idx = d
    else:
        return X

    return np.dot(U[:,:idx], np.diag(Sigma[:idx])), sum_/summation


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = get_args()
    if args.forhomework:
        X, _ = transform_data(args.moviesf, args.ratingf, args.usersf)
    else:
        X = load_data(args.fn, 'rating').values

    print X.shape

    start = time.time()
    X_transformed, retained_variance = dim_reduction_svd(X, args.d, args.v)
    print X_transformed.shape, retained_variance
    print 'kmeans time: %f' % (time.time() - start)


if __name__ == "__main__":
    main()
