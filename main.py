# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import csv
import config
import logging
import argparse
from search import CacheSearch, CWS
from dp import DPW_Cache, nmf_optimization, learn_dpwc


#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    cws = CWS(config.key)
    cache_ws = CacheSearch(cws, 'cache')

    dpw_cache = DPW_Cache(args.n, cache_ws)
    dpw_cache_nmf = {}

    reader = csv.reader(args.d, delimiter=',')
    #reader = [('monk', 'slave')]
    for row in reader:
        word_a = row[0]
        word_b = row[1]

        #word_a_neighborhood = extract_neighborhood(word_a, cache_ws, args.n)
        #word_b_neighborhood = extract_neighborhood(word_b, cache_ws, args.n)
        #logger.info('%s (%s)', word_a, len(word_a_neighborhood))
        #logger.info('%s (%s)', word_b, len(word_b_neighborhood))
        dpw_a = dpw_cache[word_a] #DPW(word_a, word_a_neighborhood)
        logger.info(dpw_a)
        dpw_b = dpw_cache[word_b] #DPW(word_b, word_b_neighborhood)
        logger.info(dpw_b)
        score_1 = dpw_a.similarity(dpw_b)

        #logger.info(dpw_cache)

        '''if dpw_a.word[1] not in dpw_cache_nmf:
            dpw_a = nmf_optimization(dpw_a, args.k, dpw_cache)
            dpw_cache_nmf[dpw_a.word[1]] = dpw_a
        else:
            dpw_a = dpw_cache_nmf[dpw_a.word[1]]

        if dpw_b.word[1] not in dpw_cache_nmf:
            dpw_b = nmf_optimization(dpw_b, args.k, dpw_cache)
            dpw_cache_nmf[dpw_b.word[1]] = dpw_b
        else:
            dpw_b = dpw_cache_nmf[dpw_b.word[1]]

        logger.info(dpw_a)
        logger.info(dpw_b)
        score_2 = dpw_a.similarity(dpw_b)'''

        dpwc_a = learn_dpwc(dpw_a, dpw_cache)
        logger.info(dpwc_a)
        dpwc_b = learn_dpwc(dpw_b, dpw_cache)
        logger.info(dpwc_b)
        

        score_2 = dpwc_a.similarity(dpwc_b)

        print(f'{word_a} {word_b} {score_1} {score_2}')
        #input('wait.................')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground')
    parser.add_argument('-d', type=argparse.FileType('r'), required=True, help='dataset file (csv)')
    parser.add_argument('-n', type=int, help='neighborhood size', default=3)
    parser.add_argument('-k', type=int, help='NMF dinamic k', default=2)
    #parser.add_argument('--r2', type=float, help='R2', )
    #parser.add_argument('-t', type=float, help='Sensitivity', default=1.0)
    #parser.add_argument('-r', type=bool, help='Ranking relative', default=True)
    #parser.add_argument('-m', type=Method, choices=list(Method), default='kneedle')
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)
