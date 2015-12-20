#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import math
import numpy as np
cimport numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from utilities import *


DTYPE = np.int
ctypedef np.int_t DTYPE_t


cdef init_kmeans(X, k):
    """TODO: Docstring for init_kmeans.
    :returns: TODO

    """
    randidx = np.random.permutation(10)
    C = X[randidx[:k],:]
    return C


cdef update_C(X, Y, C):
    """TODO: Docstring for update_C.
    :returns: TODO

    """
    cdef int idx
    for idx in range(C.shape[0]):
        points = X[Y==idx,:]
        unzipped = zip(*points)
        centroid_coords = [math.fsum(d_list)/points.shape[0]
                           for d_list in unzipped]
        C[idx] = centroid_coords
    return C


cdef update_Cx(X, Y, C):
    """TODO: Docstring for update_Cx.

    :X: TODO
    :C: TODO
    :k: TODO
    :returns: TODO

    """
    cdef int idx
    if Y is None:
        Y = np.zeros(X.shape[0])
    for idx in range(X.shape[0]):
        Y[idx] = np.argmin(pairwise_distances(C, X[idx,:].reshape(1,-1)))

    return Y


cdef cost_func(X, Y, C):
    """TODO: Docstring for cost_func.
    :returns: TODO

    """
    cdef int idx
    cdef double cost = 0
    for idx in range(C.shape[0]):
        points = X[Y==idx,:]
        cost += math.fsum(pairwise_distances(points, C[idx,:].reshape(1,-1)))
    return cost


cdef kmeans(X, k, max_iter=100):
    """TODO: Docstring for kmeans.
    :returns: TODO

    """
    C = init_kmeans(X, k)
    Y = update_Cx(X, None, C)
    print 'initial cost: %f' % (cost_func(X, Y, C))

    cdef int idx
    for idx in range(max_iter):
        Y = update_Cx(X, Y, C)
        C = update_C(X, Y, C)
        print 'iteration: %d, cost: %f' % (idx, cost_func(X, Y, C))

    print cost_func(X, Y, C)

    return X, Y, C


cdef transform_data(moviesf, ratingf, usersf):
    """TODO: Docstring for transform_data.
    :returns: TODO

    """
    movies = load_data(moviesf, 'movies').values
    rating = load_data(ratingf, 'rating').values
    users = load_data(usersf, 'users').values

    # TODO missing values
    cdef np.ndarray[DTYPE_t, ndim=2] X
    X = np.full((users[-1][0], movies[-1][0],), 0, dtype=DTYPE)

    for rate in rating:
        X[rate[0]-1][rate[1]-1] = rate[2]

    return X


cdef plot_2d(X, Y, C):
    """TODO: Docstring for plot_2d.
    :returns: TODO

    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)

    print 'projected: sample: %s, feature: %s'\
            % (projected.shape[0], projected.shape[1])

    all_scatter = []
    colors = cm.rainbow(np.linspace(0, 1, len(C)), alpha=0.5)
    for i in range(len(C)):
        points = projected[Y==i,:]
        cur = plt.scatter(points[:,0], points[:,1], color=colors[i],
                          edgecolor='k', lw=0.6,
                          vmin=0, vmax=len(C))
        all_scatter.append(cur)
    # plt.legend(all_scatter, C,
    #            loc='lower left', scatterpoints=1)
    plt.clim(-0.5, 9.5)
    plt.savefig('isomap2d', dpi=500)
    plt.show()


# TODO can simply change to pca.
cdef plot_3d(X, Y, C):
    """TODO: Docstring for plot_3d.
    :returns: TODO

    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pca = PCA(n_components=3)
    projected = pca.fit_transform(X)

    print 'projected: sample: %s, feature: %s'\
            % (projected.shape[0], projected.shape[1])

    all_scatter = []
    colors = cm.rainbow(np.linspace(0, 1, len(C)), alpha=0.5)
    for i in range(len(C)):
        points = projected[Y==i,:]
        cur = ax.scatter(points[:,0], points[:,1], points[:,2],
                          color=colors[i], edgecolor='k', lw=0.1,
                          vmin=0, vmax=len(C))
        all_scatter.append(cur)
    # ax.legend(all_scatter, C,
    #            loc='lower left', scatterpoints=1)
    plt.savefig('isomap3d', dpi=500)
    plt.show()

    return True


cdef plot_kmeans(X, Y, C, ndim):
    """TODO: Docstring for plot_kmeans.
    :returns: TODO

    """
    print 'Plotting.'
    
    if ndim == 2:
        plot_2d(X, Y, C)
    elif ndim == 3:
        plot_3d(X, Y, C)


def main(args):
    """TODO: Docstring for main.
    :returns: TODO

    """
    if args.forhomework:
        X = transform_data(args.moviesf, args.ratingf, args.usersf)
    else:
        X = load_data(args.fn, 'rating').values

    print X.shape

    start = time.time()
    X, Y, C = kmeans(X, args.k, args.max_iter)
    print 'kmeans time: %f' % (time.time() - start)

    plot_kmeans(X, Y, C, args.ndim)
