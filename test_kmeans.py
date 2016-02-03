#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import argparse
import numpy as np
from math import sqrt
from datetime import datetime
from scipy.spatial.distance import cosine
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from shieh_kmeans import kmeans
from shieh_utils import *
from sklearn.cluster import KMeans


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
                        help='Preserved Dimension.', dest='d')
    parser.add_argument('-v', action='store', type=float, default=None,
                        help='Preserved Variance.', dest='v')
    parser.add_argument('--kx', action='store', type=int,
                        help='The number of cluster of user.', dest='k_x')
    parser.add_argument('--ky', action='store', type=int,
                        help='The number of cluster of item.', dest='k_y')
    parser.add_argument('--cfk', action='store', type=int, default=20,
                        help='K nearest neighbors.', dest='cfk')
    parser.add_argument('--numbin', action='store', type=int, default=None,
                        help='The number of Item Bins.', dest='numbin')
    parser.add_argument('--alg', action='store', default='baseline',
                        help='The name of the algorithm.', dest='alg')
    parser.add_argument('--scheme', action='store', default='item',
                        help='The scheme of weight computation.', dest='scheme')
    parser.add_argument('--fn', action='store',
                        help='The name of the file.', dest='fn')
    parser.add_argument('--movies', action='store',
                        help='The name of the movies file.', dest='moviesf')
    parser.add_argument('--rating', action='store',
                        help='The name of the rating file.', dest='ratingf')
    parser.add_argument('--users', action='store',
                        help='The name of the users file.', dest='usersf')
    parser.add_argument('--user2user', action='store_false',
                        help='Use user-user.', dest='item2item')
    parser.add_argument('--testit', action='store_true',
                        help='Testing model.', dest='testit')
    parser.add_argument('--svd', action='store_true',
                        help='SVDNeighbourhood.', dest='svd')
    parser.add_argument('--forhomework', action='store_true',
                        help='Homework model.', dest='forhomework')

    return parser.parse_args()


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = get_args()
    X_train, X_test,\
    rating, Bins = transform_data(args.moviesf, args.ratingf,
                                  args.usersf, True, args.numbin)

    k = 10

    scikmeans = KMeans(n_clusters=k)

    print 'Begin'
    y_true = scikmeans.fit_predict(X_train)
    print 'scikit-learn finished'
    _, y_pred, _ = kmeans(X_train, k, 8)
    print 'shieh-kmeans finished'

    print 'mutual info: %f' % (v_measure_score(y_true, y_pred))
    print 'homogeneity: %f' % (homogeneity_score(y_true, y_pred))
    print 'completenes: %f' % (completeness_score(y_true, y_pred))


if __name__ == "__main__":
    main()
