#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import heapq
import argparse
import operator
import numpy as np
from math import sqrt
from datetime import datetime
from scipy.linalg import svd
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from shieh_utils import *
from shieh_kmeans import kmeans
from shieh_svd import dim_reduction_svd


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
    parser.add_argument('-k', action='store', type=int, default=20,
                        help='The number of cluster.', dest='k')
    parser.add_argument('--cfk', action='store', type=int, default=20,
                        help='K nearest neighbors.', dest='cfk')
    parser.add_argument('--numbin', action='store', type=int, default=None,
                        help='The number of Item Bins.', dest='numbin')
    parser.add_argument('--alg', action='store', default='baseline',
                        help='The name of the algorithm.', dest='alg')
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


def cal_sim(a, b, c, ref):
    """TODO: Docstring for cal_sim.
    :returns: TODO

    """
    if c[ref] == 0:
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


def get_sims(X, user, item, k, item2item=True, reduced=None):
    """TODO: Docstring for get_sims.
    :returns: TODO

    """
    # TODO itself
    if reduced is not None:
        comp = reduced
    else:
        comp = X

    if item2item:
        a = comp[:,item]
        X = X.T
        comp = comp.T
        ref = user
    else:
        a = comp[user,:]
        ref = item

    tar = enumerate([cal_sim(a, comp[idx,:], X[idx,:], ref)
                     for idx in range(X.shape[0])])

    neighbors = zip(*heapq.nlargest(k, tar, key=operator.itemgetter(1)))
    return np.array(neighbors[0]), np.array(neighbors[1])


def get_wgts(user, item, X, k, item2item=True):
    """TODO: Docstring for get_wgts.
    :returns: TODO

    """
    pass


def baseline_estimator(X, user, item, k, Mu):
    """TODO: Docstring for baseline_estimator.

    :arg1: TODO
    :returns: TODO

    """
    items = X[:,item]
    items = items[items!=0]
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)
    if items.shape[0] == 0:
        return user_mean
    else:
        bxi = user_mean + (np.mean(items) - Mu)
        return bxi


def neighborhood_estimator(X, user, item, k, Mu, item2item=True,
                           scheme='item', reduced=None):
    """TODO: Docstring for neighborhood_estimator.

    :scheme: 'item' or 'regression'
    :returns: TODO

    """
    items = X[:,item]
    items = items[items!=0]
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)
    if items.shape[0] == 0:
        return user_mean

    indices, sims = get_sims(X, user, item, k, item2item, reduced)

    if item2item:
        bxi = user_mean + (np.mean(items) - Mu)
        multi_items = X[:,indices]

        return bxi + np.sum(sims*((X[user, indices]-user_mean)-\
                                  np.average(multi_items, axis=0,
                                             weights=multi_items.astype(bool))+\
                                  Mu))/np.sum(sims)
    else:
        return np.sum(sims*X[indices, item])/np.sum(sims)


def item_bias_t(item, items, Bin, Mu):
    """TODO: Docstring for item_bias_t.
    :returns: TODO

    """
    item_mean = np.mean(items)
    bi = item_mean - Mu
    bit = Bin[1].get(item+1, item_mean) - Bin[2]
    return bi + bit


def user_bias_t(rates, t, tu, gamma, Mu, user_mean, time_mean):
    """TODO: Docstring for user_bias_t.
    :returns: TODO

    """
    group = rates.groupby('Timestamp').Rating.mean()
    btls = (group.loc[tu] - time_mean.loc[tu]).values
    bu = user_mean - Mu
    tmp = np.exp(-gamma*np.abs((t-tu).astype('timedelta64[D]').values))
    devu = np.sum(tmp*btls)/np.sum(tmp)
    # print 't:', t
    # print 'tu:', tu
    # print 't-tu:', t-tu
    # print 'btls:', btls
    # print 'bu:', bu
    # print 'tmp:', tmp
    # print 'devu:', devu
    return bu + devu + 0


def spline_plus(X, rating, user, item, t, Mu, gamma, Bins, time_mean):
    """TODO: Docstring for spline_plus.
    :returns: TODO

    """
    items = X[:,item]
    items = items[items!=0]
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)
    if items.shape[0] == 0:
        return user_mean
    else:
        rates = rating[rating.UserID==user+1]
        timestamps = rates.Timestamp
        total = timestamps.shape[0]
        ku = total ** 0.25
        step = int(total/ku)
        tu = timestamps.iloc[[i for i in range(0, total, step)]]
        return Mu + user_bias_t(rates, t, tu, gamma, Mu, user_mean, time_mean) +\
                item_bias_t(item, items, find_bin(Bins,t), Mu)


def svd_pred(X, user, item, svd1, svd2):
    """TODO: Docstring for svd.
    :returns: TODO

    """
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)
    return user_mean + np.dot(svd1[user,:], svd2[:,item])


def rateit(X, user, item, minv, maxv, k,
           Mu=None, item2item=True, algorithm='baseline',
           t=None, gamma=None, Bins=None,
           rating=None, time_mean=None, svd1=None, svd2=None):
    """TODO: Docstring for rateit.
    :returns: TODO

    """
    if not Mu:
        Mu = np.mean(X[X!=0])

    if algorithm == 'baseline':
        res = baseline_estimator(X, user, item, k, Mu)
    elif algorithm == 'neighborhood':
        res = neighborhood_estimator(X, user, item, k, Mu,
                                     item2item, 'item', svd1)
    elif algorithm == 'temporal':
        res = spline_plus(X, rating, user, item, t, Mu, gamma, Bins, time_mean)
    elif algorithm == 'svd':
        res = svd_pred(X, user, item, svd1, svd2)
    else:
        print 'Unsupported algorithm: %s' % (algorithm)
        exit(0)

    if np.isnan(res):
        res = 0

    if res > maxv:
        res = maxv
    elif res < minv:
        res = minv

    return res


def testing(X_train, X_test, k, item2item=True, algorithm='baseline',
            gamma=None, Bins=None, rating=None,
            time_mean=None, svd1=None, svd2=None):
    """TODO: Docstring for test_for_temproal.

    :arg1: TODO
    :returns: TODO

    """
    num_sample = 1000
    y_true = np.zeros((num_sample,1))
    y_pred = np.zeros((num_sample,1))
    Mu = np.mean(X_train[X_train!=0])

    for idx in range(num_sample):
        if idx % 100 == 0:
            print 'Processing sample: %d' % (idx)
        rate = X_test[idx]
        t = datetime.fromtimestamp(rate[3]).date()
        y_true[idx] = rate[2]
        y_pred[idx] = rateit(X_train, rate[0]-1, rate[1]-1, 1, 5,
                             k, Mu, item2item, algorithm, t, gamma,
                             Bins, rating, time_mean, svd1, svd2)

    print cal_rmse(y_true, y_pred)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = get_args()
    if args.forhomework:
        start = time.time()
        X_train, X_test,\
        rating, Bins = transform_data(args.moviesf, args.ratingf,
                                      args.usersf, True, args.numbin)
        print 'loading time: %f' % (time.time() - start)
    else:
        X = load_data(args.fn, 'rating').values

    print X_train.shape, X_test.shape

    svd1 = None
    svd2 = None
    if args.alg == 'svd':
        densed = impute(X_train)
        normalized = normalize(densed, False)
        Uk, Sk, Vk, ratio = dim_reduction_svd(normalized, args.d, args.v, False)
        root_Sk = np.sqrt(Sk)
        us = np.dot(Uk, root_Sk)
        sv = np.dot(root_Sk, Vk)
        svd1 = us
        svd2 = sv
        del normalized
    elif args.svd:
        X_train
        if args.item2item:
            Uk, Sk, Vk, ratio = dim_reduction_svd(X_train.T, args.d, args.v, False)
            root_Sk = np.sqrt(Sk)
            us = np.dot(Uk, root_Sk)
            svd1 = us.T
        else:
            Uk, Sk, Vk, ratio = dim_reduction_svd(X_train, args.d, args.v, False)
            root_Sk = np.sqrt(Sk)
            us = np.dot(Uk, root_Sk)
            svd1 = us
        print ratio

    time_mean = rating.groupby('Timestamp').Rating.mean()
    start = time.time()
    if args.testit:
        testing(X_train, X_test, args.cfk, args.item2item,
                args.alg, 0.3, Bins, rating, time_mean, svd1, svd2)
    else:
        t = datetime.fromtimestamp(978824268).date()
        train_mu = np.mean(X_train[X_train!=0])
        res = rateit(X_train, 0, 0, 1, 5, args.cfk, train_mu, args.item2item, args.alg, t, 0.3, Bins, rating, time_mean, svd1, sdv2)
        print X_train[0,0], res
    print 'recsys time: %f' % (time.time() - start)


if __name__ == "__main__":
    main()
