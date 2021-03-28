# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import csv
import config
import logging
import argparse
from pathlib import Path
from search import CacheSearch, CWS
from dp import DPW_Cache, nmf_optimization, learn_dpwc, latent_analysis
from utils import progress_bar


#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    cws = CWS(config.key)
    cache_ws = CacheSearch(cws, config.cache)

    dpw_cache     = DPW_Cache(args.n, cache_ws)
    dpw_cache_nmf = {}
    dpwc_cache    = {}
    
    filename = Path(args.d).resolve().stem
    output_filename = f'{filename}_{args.n}_{args.k}.csv'

    count = 0

    header = ['raw', 'nmf', 'dpwc', 'dpwc_nmf', 'dpwc_fuzzy', 'dpwc_fuzzy_nmf']

    with open(args.d, 'r') as in_csv, open(output_filename , 'w') as out_csv:
        reader = csv.reader(in_csv, delimiter=',')
        writer = csv.writer(out_csv, delimiter=' ')
        # write the header
        writer.writerow(header)
        for row in reader:
            word_a = row[0]
            word_b = row[1]

            progress_bar(count, 30, status=f'({word_a:12}, {word_b:12})')

            # DPW RAW
            dpw_a = dpw_cache[word_a] 
            logger.info(dpw_a)
            dpw_b = dpw_cache[word_b] 
            logger.info(dpw_b)
            score_dpw = dpw_a.similarity(dpw_b)

            # Create Latent Features
            Va, Vra = None, None #latent_analysis(dpw_a, args.k, dpw_cache)
            Vb, Vrb = None, None #latent_analysis(dpw_b, args.k, dpw_cache)

            # DPW with Latent Features
            if dpw_a.word[1] not in dpw_cache_nmf:
                Va, Vra = latent_analysis(dpw_a, args.k, dpw_cache)
                dpw_a = nmf_optimization(dpw_a, Vra)
                dpw_cache_nmf[dpw_a.word[1]] = dpw_a
            else:
                dpw_a = dpw_cache_nmf[dpw_a.word[1]]

            if dpw_b.word[1] not in dpw_cache_nmf:
                Vb, Vrb = latent_analysis(dpw_b, args.k, dpw_cache)
                dpw_b = nmf_optimization(dpw_b, Vrb)
                dpw_cache_nmf[dpw_b.word[1]] = dpw_b
            else:
                dpw_b = dpw_cache_nmf[dpw_b.word[1]]

            score_dpw_nmf = dpw_a.similarity(dpw_b)

            # DPWC and variations
            if dpw_a.word[1] not in dpwc_cache:
                dpwc_a = learn_dpwc(dpw_a, Va, Vra)
                dpwc_cache[dpw_a.word[1]] = dpwc_a
            else:
                dpwc_a = dpwc_cache[dpw_a.word[1]]

            if dpw_b.word[1] not in dpwc_cache:
                dpwc_b = learn_dpwc(dpw_b, Vb, Vrb)
                dpwc_cache[dpw_b.word[1]] = dpwc_b
            else:
                dpwc_b = dpwc_cache[dpw_b.word[1]]

            # Unpack dpwc version
            dpwc_a_kmeans, dpwc_a_nmf_kmeans, dpwc_a_fuzzy, dpwc_a_nmf_fuzzy = dpwc_a
            dpwc_b_kmeans, dpwc_b_nmf_kmeans, dpwc_b_fuzzy, dpwc_b_nmf_fuzzy = dpwc_b

            logger.info(dpwc_a_fuzzy)
            logger.info(dpwc_b_fuzzy)

            score_dpwc_kmeans = dpwc_a_kmeans.similarity(dpwc_b_kmeans)
            score_dpwc_nmf_kmeans = dpwc_a_nmf_kmeans.similarity(dpwc_b_nmf_kmeans)
            score_dpwc_fuzzy = dpwc_a_fuzzy.similarity(dpwc_b_fuzzy)
            score_dpwc_nmf_fuzzy = dpwc_a_nmf_fuzzy.similarity(dpwc_b_nmf_fuzzy)
            
            #print(f'{word_a} {word_b} {score_dpw} {score_dpw_nmf} {score_dpwc_fuzzy} {score_dpwc_nmf_fuzzy}')
            fields = [score_dpw, score_dpw_nmf, score_dpwc_kmeans, score_dpwc_nmf_kmeans, score_dpwc_fuzzy, score_dpwc_nmf_fuzzy]
            writer.writerow(fields)
            count += 1
    progress_bar(count, 30, status=f'({word_a:12}, {word_b:12})')
    print()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground')
    parser.add_argument('-d', type=str, required=True, help='dataset file (csv)')
    parser.add_argument('-n', type=int, help='neighborhood size', default=3)
    parser.add_argument('-k', type=int, help='NMF dinamic k', default=2)
    args = parser.parse_args()
    main(args)
