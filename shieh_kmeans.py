#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import math
import argparse
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from shieh_utils import *


VERSION = 'v4.0.0'
USAGE = '''usage: %(prog)s [options] arg1 arg2'''


def get_args():
    """TODO: Docstring for get_args.
    :returns: TODO

    """
    parser = argparse.ArgumentParser(usage=USAGE)

    parser.add_argument('--version', action='version',
                        help='Version.', version=VERSION)
    parser.add_argument('-k', action='store', type=int, default=2,
                        help='The name of Cluster.', dest='k')
    parser.add_argument('-i', action='store', type=int, default=10,
                        help='Maximum iteration.', dest='max_iter')
    parser.add_argument('--ndim', action='store', type=int, default=2,
                        help='Visualization dimention.', dest='ndim')
    parser.add_argument('--fn', action='store',
                        help='The name of the file.', dest='fn')
    parser.add_argument('--movies', action='store',
                        help='The name of the movies file.', dest='moviesf')
    parser.add_argument('--rating', action='store',
                        help='The name of the rating file.', dest='ratingf')
    parser.add_argument('--users', action='store',
                        help='The name of the users file.', dest='usersf')
    parser.add_argument('--plot', action='store_true',
                        help='Plot.', dest='plot')
    parser.add_argument('--forhomework', action='store_true',
                        help='Homework model.', dest='forhomework')

    return parser.parse_args()


def init_kmeans(X, k):
    """TODO: Docstring for init_kmeans.
    :returns: TODO

    """
    randidx = np.random.permutation(X.shape[0])
    C = X[randidx[:k],:]
    return C


def update_C(X, Y, C):
    """TODO: Docstring for update_C.
    :returns: TODO

    """
    for idx in range(C.shape[0]):
        points = X[Y==idx,:]
        if points.shape[0] == 0:
            continue
        unzipped = zip(*points)
        centroid_coords = [math.fsum(d_list)/points.shape[0]
                           for d_list in unzipped]
        C[idx] = centroid_coords
    return C


def update_Cx(X, Y, C):
    """TODO: Docstring for update_Cx.

    :X: TODO
    :C: TODO
    :k: TODO
    :returns: TODO

    """
    if Y is None:
        Y = np.zeros(X.shape[0])
    for idx in range(X.shape[0]):
        Y[idx] = np.argmin(pairwise_distances(C, X[idx,:].reshape(1,-1)))

    return Y


def cost_func(X, Y, C):
    """TODO: Docstring for cost_func.
    :returns: TODO

    """
    cost = 0
    for idx in range(C.shape[0]):
        points = X[Y==idx,:]
        if points.shape[0] == 0:
            continue
        cost += math.fsum(pairwise_distances(points, C[idx,:].reshape(1,-1)))
    return cost


def kmeans(X, k, max_iter=100):
    """TODO: Docstring for kmeans.
    :returns: TODO

    """
    C = init_kmeans(X, k)
    Y = update_Cx(X, None, C)
    print 'initial cost: %f' % (cost_func(X, Y, C))

    for idx in range(max_iter):
        Y = update_Cx(X, Y, C)
        C = update_C(X, Y, C)
        print 'iteration: %d, cost: %f' % (idx, cost_func(X, Y, C))

    print 'final cost: %f' % (cost_func(X, Y, C))

    return X, Y, C


def plot_kmeans(X, Y, C, ndim):
    """TODO: Docstring for plot_kmeans.
    :returns: TODO

    """
    print 'Plotting.'
    
    if ndim == 2:
        plot_2d(X, Y, C)
    elif ndim == 3:
        plot_3d(X, Y, C)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = get_args()
    if args.forhomework:
        X, _, _, _, = transform_data(args.moviesf, args.ratingf, args.usersf)
    else:
        X = load_data(args.fn, 'rating').values

    print X.shape

    start = time.time()
    X, Y, C = kmeans(X, args.k, args.max_iter)
    print 'kmeans time: %f' % (time.time() - start)

    if args.plot:
        plot_kmeans(X, Y, C, args.ndim)


if __name__ == "__main__":
    main()
