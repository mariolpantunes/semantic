# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
from scipy.optimize import nnls


logger = logging.getLogger(__name__)


def cost(V, W, H):
    mask = V > 0.0  # pd.DataFrame(V).notnull().values
    #logger.debug('Mask: %s', mask)
    WH = np.dot(W, H)
    WH_mask = WH[mask]
    #logger.debug('WH_mask: %s', WH_mask)
    A_mask = V[mask]
    #logger.debug('A_mask: %s', WH_mask)
    A_WH_mask = A_mask - WH_mask
    # Since now A_WH_mask is a vector, we use L2 instead of Frobenius norm for matrix
    return np.linalg.norm(A_WH_mask, 2)


def nmf_nnls(V, k, num_iter=1000):
    rows, columns = V.shape

    # Create W and H
    W = np.abs(np.random.uniform(low=0, high=1, size=(rows, k)))
    H = np.abs(np.random.uniform(low=0, high=1, size=(k, columns)))
    W = np.divide(W, k*W.max())
    H = np.divide(H, k*H.max())

    # Optimize
    #num_display_cost = max(int(num_iter/10), 1)
    for i in range(num_iter):
        if i % 2 == 0:
            # Learn H, given A and W
            for j in range(columns):
                mask_rows = V[:, j] > 0.0  # pd.Series(A[:,j]).notnull()
                H[:, j] = nnls(W[mask_rows], V[:, j][mask_rows])[0]
        else:
            for j in range(rows):
                mask_rows = V[j, :] > 0.0  # pd.Series(A[j,:]).notnull()
                W[j, :] = nnls(H.transpose()[mask_rows], V[j, :][mask_rows])[0]
        #WH = np.dot(W, H)
        #c = cost(V, W, H)
        #if i % num_display_cost == 0:
        #    logger.debug('%s: %s', i, c)

    return W, H

# regularized non-negative matrix factorization
def rwnmf(X, k, alpha=0.1, tol_fit_improvement=1e-4, tol_fit_error=1e-4, num_iter=1000):
    # applies regularized weighted nmf to matrix X with k factors
    # ||X-UV^T||
    eps = np.finfo(float).eps
    early_stop = False

    # get observations matrix W
    #W = np.isnan(X)
    #print('W')
    #print(W)
    #X[W] = 0  # set missing entries as 0
    #print(X)
    #W = ~W
    #print('~W')
    #print(W)
    W = X > 0.0

    # initialize factor matrices
    rnd = np.random.RandomState()
    U = rnd.rand(X.shape[0], k)
    U = np.maximum(U, eps)

    V = np.linalg.lstsq(U, X, rcond=None)[0].T
    V = np.maximum(V, eps)

    Xr = np.inf * np.ones(X.shape)

    for i in range(num_iter):
        # update U
        U = U * np.divide(((W * X) @ V), (W * (U @ V.T) @ V + alpha * U))
        U = np.maximum(U, eps)
        # update V
        V = V * np.divide((np.transpose(W * X) @ U),
                          (np.transpose(W * (U @ V.T)) @ U + alpha * V))
        V = np.maximum(V, eps)

        # compute the resduals
        if i % 10 == 0:
            # compute error of current approximation and improvement
            Xi = U @ V.T
            fit_error = np.linalg.norm(X - Xi, 'fro')
            fit_improvement = np.linalg.norm(Xi - Xr, 'fro')

            # update reconstruction
            Xr = np.copy(Xi)

            # check if early stop criteria is met
            if fit_error < tol_fit_error or fit_improvement < tol_fit_improvement:
                error = fit_error
                early_stop = True
                break

    if not early_stop:
        Xr = U @ V.T
        error = np.linalg.norm(X - Xr, 'fro')

    return Xr, U, V, error


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * \
                            (2 * eij * Q[k][j] - beta * P[i][k])

                        Q[k][j] = Q[k][j] + alpha * \
                            (2 * eij * P[i][k] - beta * Q[k][j])
        
        eR = np.dot(P, Q)
        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T



#V = np.array([[1,0,0.3],[0,1,.2],[0,.3,1]])
#print(V)
#W,H = nmf_nnls(V, 3)
#VR = np.dot(W,H)
#print(VR)
#print(f'nnls ({cost(V, W, H)})')

#print()
#V[V == 0] = np.nan
#Xr, U, V, error = rwnmf(V, 3)
#print(Xr)
#print(f'rwnmf ({error})')
#print(f'rwnmf ({cost(V, W, H.T)})')