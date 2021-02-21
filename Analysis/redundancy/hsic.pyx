#cython: language_level=3
"""https://github.com/felix-laumann/MMD_HSIC_non-stationary/blob/master/HSIC_non-stationary.ipynb

From https://arxiv.org/pdf/2010.00271.pdf:
To test for independence, the null hypothesis is H0:PXY=PXPY. We assume we observe measurements x_{i,t} and y_{i,t} over temporal grids of length TX and TY in the interval I= [0,1], respectively.

"""


import numpy as np
cimport numpy as np
cimport cython
from scipy.stats import gamma
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

cpdef cython.numeric width(cython.numeric[:,:] Z):
    dist_mat = pairwise_distances(Z, metric='euclidean')
    return np.median(dist_mat[dist_mat > 0])


cpdef HSIC_permutations(cython.numeric[:,:] X, cython.numeric[:,:] Y, float alpha, cython.numeric width_X = -1, cython.numeric width_Y = -1, int shuffle = 100):  # set widths to -1 for median heuristics

    m = X.shape[0]

    # median heuristics for kernel width
    if width_X == -1.:
        width_X = width(X)
    if width_Y == -1.:
        width_Y = width(Y)

    K = pairwise_kernels(X, X, metric='rbf', gamma=0.5 / (width_X ** 2))
    L = pairwise_kernels(Y, Y, metric='rbf', gamma=0.5 / (width_Y ** 2))
    H = np.eye(m) - (1 / m) * (np.ones((m, m)))
    # ...to center K
    K_c = H @ K @ H

    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    KL = np.dot(K, L)
    HSIC_arr = np.zeros(shuffle)
    for sh in range(shuffle):
        index_perm = np.random.permutation(L.shape[0])
        L_perm = L[np.ix_(index_perm, index_perm)]
        HSIC_arr[sh] = np.trace(np.dot(K, L_perm)) / (m * (m - 3)) + np.sum(K) * np.sum(L_perm) / (
                    m * (m - 3) * (m - 1) * (m - 2)) - 2 * np.sum(np.dot(K, L_perm)) / (m * (m - 3) * (m - 2))
    HSIC_arr_sort = np.sort(HSIC_arr)
    # stat, threshold
    return np.trace(KL) / (m * (m - 3)) + np.sum(K) * np.sum(L) / (m * (m - 3) * (m - 1) * (m - 2)) - 2 * np.sum(KL) / (
            m * (m - 3) * (m - 2)),  HSIC_arr_sort[round((1 - alpha) * shuffle)]


cpdef HSIC_gamma(cython.numeric[:,:] X, cython.numeric[:,:] Y, float alpha, cython.numeric width_X = -1, cython.numeric width_Y = -1):  # set widths to -1 for median heuristics

    m = X.shape[0]
    if width_X == -1:
        width_X = width(X)
    if width_Y == -1:
        width_Y = width(Y)

    K = pairwise_kernels(X, X, metric='rbf', gamma=0.5 / (width_X ** 2))
    L = pairwise_kernels(Y, Y, metric='rbf', gamma=0.5 / (width_Y ** 2))
    H = np.eye(m) - (1 / m) * (np.ones((m, m)))

    K_c = H @ K @ H
    L_c = H @ L @ H

    # unbiased statistics
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    KL = np.dot(K, L)

    vHSIC = np.power(1 / 6 * KL, 2)
    vaHSIC = 1 / (m * (m - 1)) * (np.sum(vHSIC) - np.trace(vHSIC))
    varHSIC = 72 * (m - 4) * (m - 5) / (m * (m - 1) * (m - 2) * (m - 3)) * vaHSIC  # variance under H0
    bone = np.ones(m)
    mu_X = 1 / (m * (m - 1)) * bone @ (K @ bone)
    mu_Y = 1 / (m * (m - 1)) * bone @ (L @ bone)
    mHSIC = 1 / m * (1 + mu_X * mu_Y - mu_X - mu_Y)  # mean under H0
    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * m / mHSIC
    # stat, threshold
    return np.trace(KL) / (m * (m - 3)) + np.sum(K) * np.sum(L) / (m * (m - 1) * (m - 2) * (m - 3)) - 2. * np.sum(
        KL) / (m * (m - 3) * (m - 2)), gamma.ppf(1 - alpha, al, scale=bet)

cpdef list time_sampler( X, time_samples,  max_time = 1000):
    """
    For a list of runs of the same process, returns array of each at specified times.
    Samples using binary search algorithm (https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html).
    
    :param X: list of time-indexed (sorted) data of the same process with the same max running time.
    :param time_samples: list of times to sample at.
    :param max_time: maximum allotted time per simulation.
    :return: data at sampled times.
    """
    cdef list ret = []
    for time in time_samples:
        data_slice = []
        for proc in X:
            time_list = list(proc.keys())
            insertion_point = np.searchsorted(time_list, time) # a[i-1] < v <= a[i] via binary search algo
            if time_list[insertion_point] != time:
                insertion_point = time_list[insertion_point - 1] # Cadlag
            data_slice.append(proc[insertion_point])
        ret.append(data_slice)
    return ret