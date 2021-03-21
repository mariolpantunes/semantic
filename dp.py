# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
from typing import List, Dict
import nltk
import numpy as np


logger = logging.getLogger(__name__)


class DPW:
    def __init__(self, word, neighborhood: List):
        ps = nltk.stem.PorterStemmer()
        self.word = (ps.stem(word), word)
        # reduce neighborhood using the stem transformation
        self.neighborhood = {}
        self.names = {}
        t = max_value = 0.0
        for k,v in neighborhood:
            stem = ps.stem(k)
            if stem not in self.neighborhood:
                self.neighborhood[stem] = 0
                self.names[stem] = k
            elif len(self.names[stem]) > len(k):
                self.names[stem] = k
            self.neighborhood[stem] += v
            t += v
            if v > max_value:
                max_value = v
        
        # add itself if not in the profile
        if self.word[0] not in self.neighborhood:
            self.neighborhood[self.word[0]] = max_value
        self.names[self.word[0]] = word
        
        # normalize the profile
        for k in self.neighborhood:
            self.neighborhood[k] /= t
    
    def get_names(self):
        return [(k, v) for k, v in self.names.items()] 
    
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
    
    def __str__(self):
        return f'Profile: {self.word}\nNeighborhood: {self.neighborhood}\nNames: {self.names}'


def nmf_optimization(dpw: DPW, dpw_cache: Dict):
    names = dpw.get_names()
    logger.debug(names)

    # Create a square matrix
    V = np.zeros(shape=(len(names), len(names)))

    # Fill the matrix
    #for 