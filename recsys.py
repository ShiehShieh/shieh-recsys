#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import heapq
import argparse
import operator
import numpy as np
from math import sqrt
from datetime import datetime
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from shieh_kmeans import kmeans
from shieh_svd import dim_reduction_svd
from shieh_utils import transform_data, load_data, impute, normalize, find_bin


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


def check_range(res, minv=1, maxv=5):
    """TODO: Docstring for check_range.
    :returns: TODO

    """
    if res > maxv:
        res = maxv
    elif res < minv:
        res = minv
    return res


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


def baseline_pred(X, user, item, Mu, **kargs):
    """TODO: Docstring for baseline_pred.

    :arg1: TODO
    :returns: TODO

    """
    items = X[:,item]
    items = items[items!=0]
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)

    if items.shape[0] == 0:
        res = user_mean
    else:
        bxi = user_mean + (np.mean(items) - Mu)
        res = bxi

    return check_range(res)


def neighborhood_pred(X, user, item, k, Mu, item2item=True,
                      scheme='item', reduced=None, **kargs):
    """TODO: Docstring for neighborhood_pred.

    :scheme: 'item' or 'regression'
    :returns: TODO

    """
    items = X[:,item]
    items = items[items!=0]
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)
    if items.shape[0] == 0:
        return check_range(user_mean)

    indices, sims = get_sims(X, user, item, k, item2item, reduced)
    multi_items = X[:,indices]
    # TODO items dot sims = item
    if scheme == 'regression':
        clf = LinearRegression()
        clf.fit(multi_items, X[:,item])
        sims = clf.coef_

    if item2item:
        bxi = user_mean + (np.mean(items) - Mu)

        res = bxi + np.sum(sims*((X[user, indices]-user_mean)-\
                                  np.average(multi_items, axis=0,
                                             weights=multi_items.astype(bool))+\
                                  Mu))/np.sum(sims)
    else:
        res = np.sum(sims*X[indices, item])/np.sum(sims)

    return check_range(res)


def cluster_useritem(X, k_x, k_y):
    """TODO: Docstring for cluster_useritem.
    :returns: TODO

    """
    X, Y_x, C_x = kmeans(X, k_x, 6)
    _, Y_y, C_y = kmeans(X.T, k_y, 6)

    C_C = np.zeros((k_x, k_y), dtype=np.float)

    for i in range(k_x):
        for j in range(k_y):
            entries = X[Y_x==i,:][:,Y_y==j]
            nonzeros = entries[entries!=0]
            if nonzeros.shape[0] != 0:
                C_C[i,j] = np.mean(nonzeros)

    return C_C, Y_x, Y_y


def kmeans_pred(C_C, user, item, Y_x, Y_y, cfk, item2item, **kargs):
    """TODO: Docstring for kmeans_estimator.
    :returns: TODO

    """
    Y_user = Y_x[user]
    Y_item = Y_y[item]
    res = C_C[Y_user, Y_item]

    if res == 0:
        indices, sims = get_sims(C_C, Y_user, Y_item, cfk, item2item, None)
        if item2item:
            multi_items = C_C[:,indices]
            res = np.sum(sims*(np.average(multi_items, axis=0,
                                          weights=multi_items.astype(bool))))/np.sum(sims)
        else:
            multi_users = C_C[indices,:]
            res = np.sum(sims*(np.average(multi_users, axis=1,
                                          weights=multi_users.astype(bool))))/np.sum(sims)
        C_C[Y_user, Y_item] = res

    return check_range(res)


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


def spline_plus(X, user, item, t, Mu, gamma, Bins, time_mean, rating, **kargs):
    """TODO: Docstring for spline_plus.
    :returns: TODO

    """
    items = X[:,item]
    items = items[items!=0]
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)

    if items.shape[0] == 0:
        res = user_mean
    else:
        rates = rating[rating.UserID==user+1]
        timestamps = rates.Timestamp
        total = timestamps.shape[0]
        ku = total ** 0.25
        step = int(total/ku)
        tu = timestamps.iloc[[i for i in range(0, total, step)]]
        res = Mu + user_bias_t(rates, t, tu, gamma, Mu, user_mean, time_mean) +\
                item_bias_t(item, items, find_bin(Bins,t), Mu)

    return check_range(res)


def svd_pred(X, user, item, svd1, svd2, **kargs):
    """TODO: Docstring for svd.
    :returns: TODO

    """
    user_mean = np.mean(X[user,X[user,:]!=0], dtype=np.float)
    return check_range(user_mean + np.dot(svd1[user,:], svd2[:,item]))


def testing(X_train, X_test, func, **arg):
    """TODO: Docstring for test_for_temproal.

    :arg1: TODO
    :returns: TODO

    """
    num_sample = 600
    # subset = X_test[np.random.randint(X_test.shape[0],size=num_sample),:]
    subset = X_test[range(0, X_test.shape[0], X_test.shape[0]/num_sample),:]

    print 'Testing Samples: %d' % (subset.shape[0])

    y_true = np.zeros((num_sample,1))
    y_pred = np.zeros((num_sample,1))

    for idx in range(num_sample):
        if idx % 100 == 0:
            print 'Processing sample: %d' % (idx)
        rate = subset[idx]
        t = datetime.fromtimestamp(rate[3]).date()
        y_true[idx] = rate[2]
        y_pred[idx] = func(X_train, int(rate[0]-1), int(rate[1]-1), t=t, **arg)

    print cal_rmse(y_true, y_pred)


def svd_wrapper(d, v, alg, svd, item2item, X):
    """TODO: Docstring for svd_wrapper.
    :returns: TODO

    """
    svd1 = None
    svd2 = None
    ratio = None

    if alg == 'svd':
        densed = impute(X)
        normalized = normalize(densed, False)
        Uk, Sk, Vk, ratio = dim_reduction_svd(normalized, d, v, False)
        root_Sk = np.sqrt(Sk)
        us = np.dot(Uk, root_Sk)
        sv = np.dot(root_Sk, Vk)
        svd1 = us
        svd2 = sv
    elif svd:
        if item2item:
            Uk, Sk, Vk, ratio = dim_reduction_svd(X.T, d, v, False)
            root_Sk = np.sqrt(Sk)
            us = np.dot(Uk, root_Sk)
            svd1 = us.T
        else:
            Uk, Sk, Vk, ratio = dim_reduction_svd(X, d, v, False)
            root_Sk = np.sqrt(Sk)
            us = np.dot(Uk, root_Sk)
            svd1 = us

    return svd1, svd2, ratio


def baseline_env(X_train, X_test, rating, Bins, args):
    """TODO: Docstring for baseline_env.

    :arg1: TODO
    :returns: TODO

    """
    Mu = np.mean(X_train[X_train!=0])
    if args.testit:
        testing(X_train, X_test, baseline_pred, Mu=Mu)


def neighborhood_env(X_train, X_test, rating, Bins, args):
    """TODO: Docstring for _env.

    :arg1: TODO
    :returns: TODO

    """
    Mu = np.mean(X_train[X_train!=0])
    svd1, svd2, ratio = svd_wrapper(args.d, args.v, args.alg, args.svd,
                                    args.item2item, X_train)
    if args.testit:
        testing(X_train, X_test, neighborhood_pred, k=args.cfk, Mu=Mu,
                item2item=args.item2item, scheme=args.scheme, reduced=svd1)


def temporal_env(X_train, X_test, rating, Bins, args):
    """TODO: Docstring for _env.

    :arg1: TODO
    :returns: TODO

    """
    Mu = np.mean(X_train[X_train!=0])
    time_mean = rating.groupby('Timestamp').Rating.mean()
    if args.testit:
        testing(X_train, X_test, spline_plus, Mu=Mu, gamma=0.3,
                Bins=Bins, time_mean=time_mean, rating=rating)


def svd_env(X_train, X_test, rating, Bins, args):
    """TODO: Docstring for _env.

    :arg1: TODO
    :returns: TODO

    """
    svd1, svd2, ratio = svd_wrapper(args.d, args.v, args.alg, args.svd,
                                    args.item2item, X_train)
    print 'dimension: %d' % (svd1.shape[1])
    print 'ratio: %f' % (ratio)
    if args.testit:
        testing(X_train, X_test, svd_pred, svd1=svd1, svd2=svd2)


def kmeans_env(X_train, X_test, rating, Bins, args):
    """TODO: Docstring for _env.

    :arg1: TODO
    :returns: TODO

    """
    C_C, Y_x, Y_y = cluster_useritem(X_train, args.k_x, args.k_y)
    if args.testit:
        testing(C_C, X_test, kmeans_pred, Y_x=Y_x, Y_y=Y_y,
                cfk=args.cfk, item2item=args.item2item)


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

    start = time.time()
    if args.alg== 'baseline':
        baseline_env(X_train, X_test, rating, Bins, args)
    elif args.alg== 'neighborhood':
        neighborhood_env(X_train, X_test, rating, Bins, args)
    elif args.alg== 'temporal':
        temporal_env(X_train, X_test, rating, Bins, args)
    elif args.alg== 'svd':
        svd_env(X_train, X_test, rating, Bins, args)
    elif args.alg== 'kmeans':
        kmeans_env(X_train, X_test, rating, Bins, args)
    else:
        print 'Unsupported algorithm: %s' % (args.alg)
        exit(0)
    print 'recsys time: %f' % (time.time() - start)


if __name__ == "__main__":
    main()
