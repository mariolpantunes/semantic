# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
from abc import ABC, abstractmethod
from typing import List, Dict

logger = logging.getLogger(__name__)


class Corpus(ABC):
    @abstractmethod
    def get(self, term: str):
        pass


class DummyCorpus(Corpus):
    def __init__(self, dataset: Dict[str, List[str]]):
        self.dataset = dataset
    
    def get(self, term: str):
        return self.dataset[term]


class WebCorpus(Corpus):
    def __init__(self):
        pass