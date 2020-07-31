"""
Created on Dec 29 2019,
By zhixiang
"""

import os
import csv
import math
import numpy as np
import pandas as pd
import numpy.linalg as la
from scipy.sparse.linalg import eigs

from sklearn.metrics import mean_squared_error, mean_absolute_error

def mkdir_file(file_path):
    directory = os.path.dirname(file_path)
    if not  os.path.exists(directory):
        os.makedirs(directory)

def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean

    
def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError('ERROR: the size of spatial kernel must be greater than 1, but received "%s".' % Ks)


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)




def mean_absolute_percentage_error(y_true, y_pred):    
    y_true[y_true<1]=0
    idx = np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100


"""###### evaluation ######"""
def evaluation(GTS, PREDS):
    

    n_links = GTS.shape[1]
    n_inst = GTS.shape[0]
    RMSE, MAE, MAPE, FNORM, R2, VARSCORE = 0, 0, 0, 0, 0, 0
    for idx in range(n_links):
        gts = GTS[:, idx:idx+1]
        preds = PREDS[:, idx:idx+1]
        rmse = math.sqrt(mean_squared_error(gts, preds))
        mae = mean_absolute_error(gts, preds)
        mape = mean_absolute_percentage_error(gts, preds)
        F_norm = la.norm(gts - preds, 'fro') / la.norm(gts, 'fro')
        r2 = 1 - ((gts - preds) ** 2).sum() / ((gts - gts.mean()) ** 2).sum()
        var = np.var(gts - preds)
        var_score = 1 - (var) / np.var(gts)

        RMSE += rmse
        MAE += mae
        MAPE += mape
        FNORM +=F_norm
        R2 += r2
        VARSCORE += var_score
    return  RMSE/n_links, MAE/n_links, MAPE/n_links, FNORM/n_links, R2/n_links, VARSCORE/n_links
    
def evaluation0(gts, preds):
    # print('preds shape:\t', preds.shape)
    rmse = math.sqrt(mean_squared_error(gts, preds))
    mae = mean_absolute_error(gts, preds)
    mape = mean_absolute_percentage_error(gts, preds)
    F_norm = la.norm(gts - preds, 'fro') / la.norm(gts, 'fro')
    r2 = 1 - ((gts - preds) ** 2).sum() / ((gts - gts.mean()) ** 2).sum()
    var = np.var(gts - preds)
    var_score = 1 - (var) / np.var(gts)

    return rmse, mae, mape, F_norm, r2, var_score

