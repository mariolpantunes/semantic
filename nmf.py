# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
import scipy


logger = logging.getLogger(__name__)


def cost(V, W, H):
    mask = V > 0.0 #pd.DataFrame(V).notnull().values
    logger.debug('Mask: %s', mask)
    WH = np.dot(W, H)
    WH_mask = WH[mask]
    logger.debug('WH_mask: %s', WH_mask)
    A_mask = V[mask]
    logger.debug('A_mask: %s', WH_mask)
    A_WH_mask = A_mask - WH_mask
    # Since now A_WH_mask is a vector, we use L2 instead of Frobenius norm for matrix
    return np.linalg.norm(A_WH_mask, 2)


def annls(V, k, num_iter = 1000):
    rows, columns = V.shape
    
    # Create W and H
    W = np.abs(np.random.uniform(low=0, high=1, size=(rows, k)))
    H = np.abs(np.random.uniform(low=0, high=1, size=(k, columns)))
    W = np.divide(W, k*W.max())
    H = np.divide(H, k*H.max())

    logger.debug(W)
    logger.debug(H)

    # Optimize
    num_display_cost = max(int(num_iter/10), 1)
    for i in range(num_iter):
        if i%2 ==0:
            # Learn H, given A and W
            for j in range(columns):
                mask_rows =  V[:,j] > 0.0 #pd.Series(A[:,j]).notnull()
                H[:,j] = scipy.optimize.nnls(W[mask_rows], V[:,j][mask_rows])[0]
        else:
            for j in range(rows):
                mask_rows =  V[j,:] > 0.0 #pd.Series(A[j,:]).notnull()
                W[j,:] = scipy.optimize.nnls(H.transpose()[mask_rows], V[j,:][mask_rows])[0]
        WH = np.dot(W, H)
        c = cost(V, W, H)
        if i%num_display_cost==0:
            logger.debug('%s: %s', i, c)
    
    return W, H
