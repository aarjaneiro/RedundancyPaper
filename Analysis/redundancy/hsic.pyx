#cython: language_level=3
"""https://github.com/felix-laumann/MMD_HSIC_non-stationary/blob/master/HSIC_non-stationary.ipynb"""


import numpy as np
cimport numpy as np
cimport cython
import matplotlib.pyplot as plt
from scipy.stats import gamma, ortho_group
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

cpdef cython.numeric width(cython.numeric[:,:] Z):
    dist_mat = pairwise_distances(Z, metric='euclidean')
    return np.median(dist_mat[dist_mat > 0])


def HSIC_permutations(cython.numeric[:,:] X, cython.numeric[:,:] Y, float alpha, cython.numeric width_X = -1, cython.numeric width_Y = -1, int shuffle = 100):  # set widths to -1 for median heuristics

    m = X.shape[0]

    # median heuristics for kernel width
    if width_X == -1.:
        width_X = width(X)
    if width_Y == -1.:
        width_Y = width(Y)

    # compute Gram matrices
    K = pairwise_kernels(X, X, metric='rbf', gamma=0.5 / (width_X ** 2))
    L = pairwise_kernels(Y, Y, metric='rbf', gamma=0.5 / (width_Y ** 2))
    #KL = np.dot(K, L)

    # biased test statistic
    # centering matrix...
    H = np.eye(m) - (1 / m) * (np.ones((m, m)))

    # ...to center K
    K_c = H @ K @ H

    # unbiased statistic
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    KL = np.dot(K, L)

    # initiating HSIC
    HSIC_arr = np.zeros(shuffle)
    # create permutations by reshuffling L except the main diagonal
    for sh in range(shuffle):
        index_perm = np.random.permutation(L.shape[0])
        L_perm = L[np.ix_(index_perm, index_perm)]
        # biased
        #HSIC_arr[sh] = 1/(m**2) * np.sum(np.multiply(K_c.T, L_perm))
        # unbiased
        HSIC_arr[sh] = np.trace(np.dot(K, L_perm)) / (m * (m - 3)) + np.sum(K) * np.sum(L_perm) / (
                    m * (m - 3) * (m - 1) * (m - 2)) - 2 * np.sum(np.dot(K, L_perm)) / (m * (m - 3) * (m - 2))

    HSIC_arr_sort = np.sort(HSIC_arr)

    """
    if stat > threshold:
        print('H0 rejected')
    else:
        print('H0 accepted')
    """
    # stat, threshold
    return np.trace(KL) / (m * (m - 3)) + np.sum(K) * np.sum(L) / (m * (m - 3) * (m - 1) * (m - 2)) - 2 * np.sum(KL) / (
            m * (m - 3) * (m - 2)),  HSIC_arr_sort[round((1 - alpha) * shuffle)]

