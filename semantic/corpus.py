# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import semantic.search as search
from abc import ABC, abstractmethod
from typing import Dict, List

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
    def __init__(self, key: str, path: str, limit: int = 0):
        cws = search.CWS(key)
        self.cs = search.CacheSearch(cws, path, limit)

    def get(self, term: str):
        return self.cs.search(term)
