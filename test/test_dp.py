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
        dataset={'banana': ['A banana is an elongated, edible fruit – botanically a berry – produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas.'],
        'apple':['An apple is a round, edible fruit produced by an apple tree. Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found.'],
        'peach':['The peach is a deciduous tree first domesticated and cultivated in China. It bears edible juicy fruits with various characteristics, most called peaches and the glossy-skinned, non-fuzzy varieties called nectarines'],
        'elongate':['The verb elongate means "to make long or longer," and it stems from the Late Latin elongare, "to extend or prolong." When you stretch something out, especially when it\'s longer than it is wide, you can say you elongate it.'],
        'edible': ['An edible item is any item that is safe for humans to eat.'],
        'fruit': ['Fruits are the mature and ripened ovaries of flowers. The first step in fruit growth is fertilization of the carpel.'],
        'produced': ['The definition of produced is make or manufacture from components or raw materials.'],
        'produce': ['Produce generally refers to fresh fruits and vegetables intended to be eaten by humans, although other food products such as dairy products or nuts are sometimes included. In supermarkets, the term is also used to refer to the section of the store where fruit and vegetables are kept. Produce is the main product sold by greengrocers and farmers\' markets'],
        'tree': ['Tree, woody plant that regularly renews its growth (perennial)'],
        'cultivated': ['Some fields are cultivated while others lie fallow.'],
        'cultivate': ['The meaning of CULTIVATE is to prepare or prepare and use for the raising of crops; also : to loosen or break up the soil about (growing plants). How to use cultivate in a sentence.'],
        'worldwide': ['Kids Definition of worldwide extending over or involving the entire world'],
        'develops': ['The definition of develops is grow or cause to grow and become more mature, advanced, or elaborate.'],
        'develop': ['The meaning of develop is to set forth or make clear by degrees or in detail : expound. How to use develop in a sentence.'],
        'single': ['The definition of single is only one; not one of several.'],
        'ovary': ['The ovary is an organ found in the female reproductive system that produces an ovum.'],
        'plantain': ['Plantain is a starchy fruit that belongs to the banana family, specifically the genus Musa. Unlike dessert bananas, plantains are larger, firmer, and typically cooked before eating, often used in savory dishes in many tropical regions.'],
        'distinguish': ['To distinguish means to recognize or identify differences between people or things. It can also refer to making something special or notable in some way.'],
        'musa': ['Musa is one of three genera in the family Musaceae. The genus includes 83 species of flowering plants producing edible bananas and plantains, and fiber, used to make paper and cloth.'],
        'country': ['A country is a distinct part of the world, such as a state, nation, or other political entity. When referring to a specific polity, the term "country" may refer to a sovereign state, states with limited recognition, constituent country, or a dependent territory. Most sovereign states, but not all countries, are members of the United Nations.'],
        'cook': ['Cooking, also known as cookery or professionally as the culinary arts, is the art, science and craft of using heat to make food more palatable, digestible, nutritious, or safe. Cooking techniques and ingredients vary widely, from grilling food over an open fire, to using electric stoves, to baking in various types of ovens, reflecting local conditions.'],
        'call': ['The meaning of called is to give someone or something a name, or to know or address someone by a particular name.'],
        'dessert': ['Dessert is a course that concludes a meal. The course consists of sweet foods, such as cake, biscuit, ice cream and possibly a beverage such as dessert wine and liqueur. Some cultures sweeten foods that are more commonly savory to create desserts.'],
        'round': ['Round things are things that are approximately circular, cylindrical or spherical.  This is a common shape in nature whereby the forces of nature such as gravity or erosion have a tendency to shape things this way. Round shapes are also common in engineered things such as wheels and designed things such as clocks.'],
        'deciduous': ['Deciduous means "falling off at maturity" and refers to plants that lose their leaves, petals, or fruits seasonally. Learn about the causes, effects, and adaptations of deciduousness in different climates and regions.'],
        'first': ['First is the one coming, occurring, or ranking before or above all others.'],
        'various': ['Various is being more than one; several.'],
        'characteristic': ['The meaning of characteristic is being a feature that helps to distinguish a person or thing; distinctive. '],
        'variety': ['Variety is a number or collection of different things especially of a particular class'],
        'nectarine': ['Nectarine, (Prunus persica), smooth-skinned peach that is grown throughout the warmer temperate regions of both the Northern and Southern hemispheres. A genetic variant of common peaches, the nectarine was most likely domesticated in China more than 4,000 years ago.'],
        }
        cls.corpus = corpus.DummyCorpus(dataset)
        cls.terms = ['banana', 'apple', 'peach']
    
    def test_dp_00(self):
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none)
        model.fit(self.terms)
        result = model.similarity('banana', 'banana')
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_01(self):
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none)
        model.fit(self.terms)
        result = model.similarity('banana', 'apple')
        desired = 0.10690449676496976
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_02(self):
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(self.terms)
        result = model.similarity('banana', 'apple')
        desired = 0.3762209893173586
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_03(self):
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(self.terms)
        result = model.vocabulary()
        self.assertEqual(result, self.terms)

    def test_dp_04(self):
        model = dp.DPWModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(self.terms)
        result = model.similarity('banana', 'mango')
        desired = 0.48396088292426653
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_05(self):
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=False)
        model.fit(self.terms)
        result = model.similarity('banana', 'banana')
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_dp_06(self):
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=False)
        model.fit(self.terms)
        result = model.similarity('banana', 'apple')
        desired = 0.1045751633986928
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_07(self):
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(self.terms)
        result = model.similarity('banana', 'apple')
        desired = 0.08254792436106222
        self.assertAlmostEqual(result, desired, 2)
    
    def test_dp_08(self):
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(self.terms)
        result = model.vocabulary()
        self.assertEqual(result, self.terms)
    
    def test_dp_09(self):
        model = dp.DPWCModel(self.corpus, l=0, c=dp.Cutoff.none, latent=True)
        model.fit(self.terms)
        result = model.similarity('banana', 'mango')
        desired = 0.06935133872241445
        self.assertAlmostEqual(result, desired, 2)


if __name__ == '__main__':
    unittest.main()
