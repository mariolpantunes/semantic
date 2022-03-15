# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import semantic.corpus as corpus
import semantic.dp as dp


class TestDP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = {'banana': ['A banana is an elongated, edible fruit botanically a berry produced by several kinds of large herbaceous flowering plants in the genus Musa.'],
                   'apple': ['An apple is an edible fruit produced by an apple tree. Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus.'],
                   'peach': ['The peach develops from a single ovary that ripens into both a fleshy, juicy exterior that forms the edible part of the fruit and a hard interior.']}
        cls.corpus = corpus.DummyCorpus(dataset)
        cls.terms = ['banana', 'apple', 'peach']

    def test_dp_00(self):
        model = dp.DPWModel(self.corpus, l=0)
        model.fit(self.terms)
        result = model.similarity('banana', 'banana')
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_01(self):
        model = dp.DPWModel(self.corpus, l=0)
        model.fit(self.terms)
        result = model.similarity('banana', 'apple')
        desired = 0.17025130615174972
        self.assertAlmostEqual(result, desired, 2)

    def test_dpwc_00(self):
        model = dp.DPWCModel(self.corpus, l=0)
        model.fit(self.terms)
        print(model)


if __name__ == '__main__':
    unittest.main()