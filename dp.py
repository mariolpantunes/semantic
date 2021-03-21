# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
from typing import List
import nltk
import numpy as np


logger = logging.getLogger(__name__)


class DPW:
    def __init__(self, neighborhood: List):
        # reduce neighborhood using the stem transformation
        ps = nltk.stem.PorterStemmer()
        self.neighborhood = {}
        self.names = {}
        for k,v in neighborhood:
            stem = ps.stem(k)
            if stem not in self.neighborhood:
                self.neighborhood[stem] = 0
                self.names[stem] = k
            elif len(self.names[stem]) > len(k):
                self.names[stem] = k
            self.neighborhood[stem] += v
    
    def similarity(self, dpw: 'DPW') -> float:
        features_a = list(self.neighborhood.keys())
        features_b = list(dpw.neighborhood.keys())
        features = list(set(features_a + features_b))
        vector_a = []
        vector_b = []
        for f in features:
            if f in self.neighborhood:
                vector_a.append(self.neighborhood[f])
            else:
                vector_a.append(0.0)
            
            if f in dpw.neighborhood:
                vector_b.append(dpw.neighborhood[f])
            else:
                vector_b.append(0.0)
        
        a = np.array(vector_a)
        b = np.array(vector_b)

        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
