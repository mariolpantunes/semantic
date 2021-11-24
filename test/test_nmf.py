import unittest
import numpy as np
import semantic.nmf as nmf


class TestNMF(unittest.TestCase):
    def test_nmf_mu_3(self):
        X = np.array([[1,2,3], [2,1,3], [3,1,2]])
        W, H, _ = nmf.nmf_mu(X, 3)
        result = W@H
        np.testing.assert_almost_equal(result, X, decimal=3)
    
    def test_nmf_mu_2(self):
        X = np.array([[1,2,3], [2,1,3], [3,1,2]])
        W, H, _ = nmf.nmf_mu(X, 2)
        result = W@H
        np.testing.assert_almost_equal(result, X, decimal=0)
    
    def test_nmf_mu_1(self):
        X = np.array([[1,2,3], [2,1,3], [3,1,2]])
        W, H, _ = nmf.nmf_mu(X, 1)
        result = W@H
        np.testing.assert_almost_equal(result, X, decimal=0)
    
    def test_nmf_mu_missing_0(self):
        X = np.array([[5,0,1], [0,3,0], [0,5,1]])
        print(X)
        W, H, _ = nmf.nmf_mu(X, 3)
        result = W@H
        print(result)


if __name__ == '__main__':
    unittest.main()