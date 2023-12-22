'''Be the boss.
'''

import numpy as np

from numba import njit

from tools import linear_comb

def my_ridge(X_true, y_true, alpha):
    '''Single ridge fit.
    Returns weights for non-dependent assets and mean value of linear combination.
    '''
    X_offset = np.average(X_true, axis=0)
    y_offset = np.average(y_true, axis=0)
    X = X_true - X_offset
    y = y_true - y_offset
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    diag = s / (s ** 2 + alpha)
    diag[s < 1e-15] = 0
    d = np.diag(diag)
    coef_ = Vt.T @ d @ U.T @ y

    return coef_

def my_ridge_train(train, alpha, dependent_variable=-1, max_position_usd=1000):
    '''Full training pack.
    Returns weights for all assets and mean value of linear combination.
    '''
    nvar = train.shape[1]
    if nvar == 0:
        return np.zeros(0), 0
    dependent_variable %= nvar

    X_assets = np.arange(nvar) != dependent_variable
    X, y = train[:, X_assets], train[:, dependent_variable]
    X_offset = np.average(X, axis=0)
    y_offset = np.average(y, axis=0)
    
    betas = my_ridge(X, y, alpha)
    weights = np.zeros(nvar) - 1
    bool_assets = np.where(np.arange(nvar) != dependent_variable)
    weights[bool_assets] = betas
    divider = np.abs(weights).sum()
    weights = np.where(np.abs(weights)*max_position_usd/divider>=10, weights, 0)
    intercept = y_offset - np.dot(X_offset, weights[bool_assets])
    
    return weights, -intercept

def one_step_training(train_array: np.ndarray, test_size: int, std_coef: float, max_position_usd, dependent_asset: int, alpha=0, need=10, threshold = 10):
    '''Model training on specific sample.
    Returns weights, mean values and thresholds.
    '''
    weights, constant_term = my_ridge_train(train_array, alpha, dependent_asset, max_position_usd)

    weights_expanded = weights * np.ones((test_size, train_array.shape[1]))
    lc = linear_comb(train_array, weights_expanded)
    std_threshold = np.std(lc)*std_coef
    upper_threshold = constant_term + std_threshold
    lower_threshold = constant_term - std_threshold

    means = np.zeros(test_size) + constant_term
    upper_thresholds = np.zeros(test_size) + upper_threshold
    lower_thresholds = np.zeros(test_size) + lower_threshold
    return weights_expanded, means, upper_thresholds, lower_thresholds