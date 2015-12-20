#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA


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


def transform_data(moviesf, ratingf, usersf, splitit=False):
    """TODO: Docstring for transform_data.
    :returns: TODO

    """
    movies = load_data(moviesf, 'movies').values
    users = load_data(usersf, 'users').values
    rating = load_data(ratingf, 'rating')

    # TODO missing values
    X_train = np.full((users[-1][0], movies[-1][0],), 0, dtype=np.int)
    X_test = None

    if splitit:
        X_test = np.full((users[-1][0], movies[-1][0],), 0, dtype=np.int)
        for uid in users[:,0]:
            rates = rating[rating['UserID']==uid].sort_values(by='Timestamp')
            sep = rates.shape[0]*0.9
            for idx, rate in enumerate(rates.values):
                if idx <= sep:
                    X_train[rate[0]-1][rate[1]-1] = rate[2]
                else:
                    X_test[rate[0]-1][rate[1]-1] = rate[2]
    else:
        for rate in rating.values:
            X_train[rate[0]-1][rate[1]-1] = rate[2]

    return X_train, X_test


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
