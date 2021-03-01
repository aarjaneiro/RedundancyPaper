#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3, infer_types=True

"""
Methods for testing independence using the Hilbert-Schmidt information criterion.

Sources:

HSIC -- https://arxiv.org/pdf/2010.00271.pdf, https://github.com/felix-laumann/MMD_HSIC_non-stationary/blob/master/HSIC_non-stationary.ipynb

DHSIC -- https://arxiv.org/pdf/1603.00285.pdf
"""
import random
from copy import deepcopy
import numpy as np
cimport numpy as np
from scipy.stats import gamma
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

cpdef width(Z):
    dist_mat = pairwise_distances(Z, metric='euclidean')
    return np.median(dist_mat[dist_mat > 0])

cdef center_kl(X, Y, width_X, width_Y, m=None):
    if m is None:
        m = X.shape[0]
    if width_X == -1.:
        width_X = width(X)
    if width_Y == -1.:
        width_Y = width(Y)
    return center_k(X, width_X, m), center_k(Y, width_Y, m)

cdef center_k(X, width_X, m=None, metric="chi2", params=None):
    if m is None:
        m = X.shape[0]
    H = np.eye(m) - (1 / m) * (np.ones((m, m)))
    if params is None:
        K = pairwise_kernels(X, X, metric=metric, gamma=0.5)
    else:
        K = pairwise_kernels(X, X, metric=metric, gamma=params(X))
    K = H @ K @ H
    return K

cpdef HSIC_permutations(X, Y, float alpha, width_X = -1, width_Y = -1,
                        int shuffle = 100):  # set widths to -1 for median heuristics

    m = X.shape[0]
    K, L = center_kl(X, Y, width_X, width_Y, m)

    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    KL = np.dot(K, L)
    HSIC_arr = np.zeros(shuffle)
    for sh in range(shuffle):
        index_perm = np.random.permutation(L.shape[0])  # a sampled time
        L_perm = L[np.ix_(index_perm, index_perm)]
        HSIC_arr[sh] = np.trace(np.dot(K, L_perm)) / (m * (m - 3)) + np.sum(K) * np.sum(L_perm) / (
                m * (m - 3) * (m - 1) * (m - 2)) - 2 * np.sum(np.dot(K, L_perm)) / (m * (m - 3) * (m - 2))
    HSIC_arr_sort = np.sort(HSIC_arr)
    #stat, threshold
    return np.trace(KL) / (m * (m - 3)) + np.sum(K) * np.sum(L) / (m * (m - 3) * (m - 1) * (m - 2)) - 2 * np.sum(KL) / (
            m * (m - 3) * (m - 2)), HSIC_arr_sort[round((1 - alpha) * shuffle)]

cpdef HSIC_gamma(X, Y, float alpha, width_X = -1, width_Y = -1,
                 max_time=1000):  # set widths to -1 for median heuristics

    m = X.shape[0]
    K, L = center_kl(X, Y, width_X, width_Y, m)

    #unbiased statistics
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    KL = np.dot(K, L)

    vHSIC = np.power(1 / 6 * KL, 2)
    vaHSIC = 1 / (m * (m - 1)) * (np.sum(vHSIC) - np.trace(vHSIC))
    varHSIC = max_time * (m - 4) * (m - 5) / (m * (m - 1) * (m - 2) * (m - 3)) * vaHSIC  # variance under H0
    bone = np.ones(m)
    mu_X = 1 / (m * (m - 1)) * bone @ (K @ bone)
    mu_Y = 1 / (m * (m - 1)) * bone @ (L @ bone)
    mHSIC = 1 / m * (1 + mu_X * mu_Y - mu_X - mu_Y)  # mean under H0
    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * m / mHSIC
    #stat, threshold
    return np.trace(KL) / (m * (m - 3)) + np.sum(K) * np.sum(L) / (m * (m - 1) * (m - 2) * (m - 3)) - 2. * np.sum(
        KL) / (m * (m - 3) * (m - 2)), gamma.ppf(1 - alpha, al, scale=bet)

cpdef list time_sampler(X, time_samples):
    """
    For a list of runs of the same process, returns array of each at specified times.
    Samples using binary search algorithm (https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html).
    
    :param X: list of time-indexed (sorted) data of the same process with the same max running time.
    :param time_samples: list of times to sample at.
    :return: data at sampled times.
    """
    cdef list ret = []
    for time in time_samples:
        data_slice = []
        for proc in X:
            time_list = list(proc.keys())
            insertion_point = np.searchsorted(time_list, time)  # a[i-1] < v <= a[i] via binary search algo
            try:
                if time_list[insertion_point] != time and insertion_point > 0:
                    insertion_point = time_list[insertion_point - 1]  # take left
                else:
                    insertion_point = 0
                data_slice.append(proc[insertion_point])
            except IndexError as e: # Handles unexpected errors while printing cause
                print(e.__str__())
                data_slice.append(0)
                pass
        ret.append(data_slice)
    return ret

cdef dHSIC_hat(Xs):
    """https://arxiv.org/pdf/1603.00285.pdf -- see algorithm 1. Tests across d dists for independence beyond
    binary betyween all elements of Xs."""
    cdef int x_len = Xs[0].shape[0]
    #inits
    t1 = 1
    t2 = 1
    t3 = (2 / x_len)
    for x in Xs:
        K = center_k(x, width(x))
        t1 = np.multiply(t1, K)
        t2 = (1 / x_len ** 2) * t2 * np.sum(K)
        t3 = (1 / x_len) * t3 + np.sum(K, axis=0)
    return (1 / x_len ** 2) * np.sum(t1) + t2 - np.sum(t3)

cpdef dHSIC_resample_test(list Xs, int shuffle=500):
    """Resampling implementation -- see sec 4.2. of https://arxiv.org/pdf/1603.00285.pdf
     Returns stat and threshold (if possible)."""
    init = dHSIC_hat(Xs)
    locX = deepcopy(Xs) # deep copy
    cdef int hits = 0
    for i in range(shuffle):
        random.shuffle(locX)  # void shuffles
        permed = dHSIC_hat(locX)
        if permed >= init:
            hits += 1
    return (hits + 1) / (shuffle + 1)

cpdef list[float] spanning_grid_uniform(float length = 1000, float min_val = 0):
    """
    for values arranged in a grid, provides a draw from a uniform distribution for each int until max length.
    """
    cdef int max_val = int(np.round(length))
    cdef int step = int(np.round(min_val))
    cdef list ret = []
    while max_val > min_val:
        ret.append(random.uniform(0,1) + step)
        step += 1
        max_val -= 1
    return ret


