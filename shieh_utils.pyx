#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cython
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t


def load_data(fn, ftype):
    """TODO: Docstring for load_data.
    :returns: TODO

    """
    headers = {'movies': ['MovieID', 'Title', 'Genres'],
               'rating': ['UserID', 'MovieID', 'Rating', 'Timestamp'],
               'users': ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']}
    res = pd.read_csv(fn, sep='::', header=None,
                      names=headers[ftype], engine='python')
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def split2itembins(rating, numbin):
    """TODO: Docstring for split2bins.
    :returns: TODO

    """
    sorted_rating = rating.sort_values(by='Timestamp')
    rates = sorted_rating.Rating.values
    total = rates.shape[0]
    step = total/numbin
    Bins = [(datetime.fromtimestamp(sorted_rating.Timestamp.iloc[i]).date(),
             sorted_rating.iloc[i:i+step].groupby('MovieID').Rating.mean(),
             np.mean(rates[i:i+step]))
            for i in range(0, total, step)]
    return Bins


@cython.boundscheck(False)
def find_bin(list Bins, t):
    cdef:
        Py_ssize_t idx
    for idx in range(len(Bins)-1):
        if t >= Bins[idx][0] and t < Bins[idx+1][0]:
            return Bins[idx]
    return Bins[-1]


@cython.boundscheck(False)
@cython.wraparound(False)
def impute(np.ndarray[DTYPE_t, ndim=2] X, copy=True):
    cdef:
        double aver
        Py_ssize_t i, j
        np.ndarray[DTYPE_t, ndim=1] col
        np.ndarray[DTYPE_t, ndim=2] _X

    if copy:
        _X = np.array(X, dtype=DTYPE)
    else:
        _X = X

    for j in range(X.shape[1]):
        col = X[:,j]
        nonzeros = col[col!=0]

        if nonzeros.shape[0] != 0:
            aver = np.mean(nonzeros)

            for i in range(X.shape[0]):
                if X[i,j] == 0:
                    _X[i,j] = aver

    return _X


@cython.boundscheck(False)
@cython.wraparound(False)
def normalize(np.ndarray[DTYPE_t, ndim=2] X, copy=True):
    cdef:
        Py_ssize_t i, j
        double aver
        np.ndarray[DTYPE_t, ndim=1] row
        np.ndarray[DTYPE_t, ndim=2] _X

    if copy:
        _X = np.array(X, dtype=DTYPE)
    else:
        _X = X

    for i in range(X.shape[0]):
        row = X[i,:]
        nonzeros = row[row!=0]

        if nonzeros.shape[0] != 0:
            aver = np.mean(nonzeros)
            _X[i,:] = row - aver

    return _X


@cython.boundscheck(False)
def transform_data(moviesf, ratingf, usersf, splitit=False, numbin=0):
    """TODO: Docstring for transform_data.
    :returns: TODO

    """
    cdef:
        double sep
        DTYPE_t uid
        Py_ssize_t idx
        np.ndarray[DTYPE_t, ndim=1] rate
        np.ndarray[DTYPE_t, ndim=2] X_train, X_test

    movies = load_data(moviesf, 'movies').MovieID.values
    users = load_data(usersf, 'users').UserID.values
    rating = load_data(ratingf, 'rating')

    # TODO missing values
    X_train = np.full((users[-1], movies[-1],), 0, dtype=DTYPE)
    X_test = np.full((0,0,), 0, dtype=DTYPE)

    if splitit:
        X_test_list = []
        rating.sort_values(by=['UserID', 'Timestamp'], inplace=True)
        for uid in users:
            rates = rating.loc[rating.UserID==uid,:]
            sep = rates.shape[0]*0.9
            for idx, rate in enumerate(rates.values.astype(DTYPE)):
                if idx <= sep:
                    X_train[rate[0]-1][rate[1]-1] = rate[2]
                else:
                    X_test_list.append(rate.reshape(1,4))
        X_test = np.concatenate(X_test_list)
    else:
        for rate in rating.values.astype(DTYPE):
            X_train[rate[0]-1][rate[1]-1] = rate[2]

    if numbin:
        Bins = split2itembins(rating, numbin)
        rating.Timestamp = pd.to_datetime(rating.Timestamp, unit='s').dt.date
    else:
        Bins = None

    return X_train, X_test, rating, Bins


def plot_2d(X, Y, C):
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
def plot_3d(X, Y, C):
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
