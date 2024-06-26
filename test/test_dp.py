# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import nltk
import logging
import unittest
import semantic.dp as dp
import semantic.corpus as corpus

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TestDP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset={'banana': ['A banana is an elongated, edible fruit botanically a berry produced by several kinds of large herbaceous flowering plants in the genus Musa.'],
        'apple':['An apple is an edible fruit produced by an apple tree. Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus.'],
        'peach':['The peach develops from a single ovary that ripens into both a fleshy, juicy exterior that forms the edible part of the fruit and a hard interior.'],
        'elongated':['Something that is elongated is stretched out, or extended so that it is longer than usual.'],
        'edible': ['An edible item is any item that is safe for humans to eat.'],
        'fruit': ['Fruits are the mature and ripened ovaries of flowers. The first step in fruit growth is fertilization of the carpel.'],
        'produced': ['The definition of produced is make or manufacture from components or raw materials.'],
        'tree': ['Tree, woody plant that regularly renews its growth (perennial)'],
        'cultivated': ['Some fields are cultivated while others lie fallow.'],
        'worldwide': ['Kids Definition of worldwide extending over or involving the entire world'],
        'develops': ['The definition of develops is grow or cause to grow and become more mature, advanced, or elaborate.'],
        'single': ['The definition of single is only one; not one of several.'],
        'ovary': ['The ovary is an organ found in the female reproductive system that produces an ovum.']
        }
        cls.corpus = corpus.DummyCorpus(dataset)
        cls.terms = ['banana', 'apple', 'peach']
    
    def test_dp_00(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none)
        model.fit(terms)
        result = model.similarity('banana', 'banana')
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_01(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none)
        model.fit(terms)
        result = model.similarity('banana', 'apple')
        desired = 0.2581988897471611
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_02(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(terms)
        result = model.similarity('banana', 'apple')
        desired = 0.31543255181329394
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_03(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(terms)
        result = model.vocabulary()
        self.assertEqual(result, terms)

    def test_dp_04(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(terms)
        result = model.similarity('banana', 'mango')
        desired = 0.10514418393776466
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_05(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=False)
        model.fit(terms)
        result = model.similarity('banana', 'banana')
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_06(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=False)
        model.fit(terms)
        result = model.similarity('banana', 'apple')
        desired = 0.23643410852713176
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_07(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(terms)
        result = model.similarity('banana', 'apple')
        desired = 0.1934145047219262
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_08(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(terms)
        result = model.vocabulary()
        self.assertEqual(result, terms)
    
    def test_dp_09(self):
        terms = ['banana', 'apple', 'peach']
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(terms)
        result = model.similarity('banana', 'mango')
        desired = 0.0644715015739754
        self.assertAlmostEqual(result, desired, 2)


if __name__ == '__main__':
    unittest.main()
