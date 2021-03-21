# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import csv
import nltk
import config
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from search import CacheSearch, CWS
from dp import DPW, nmf_optimization
import nmf


logging.basicConfig(level=logging.DEBUG)


logger = logging.getLogger(__name__)


def extract_neighborhood(target_word, ws, n):
    snippets = ws.search(target_word)
    ps = nltk.stem.PorterStemmer()
    stem_target_word = ps.stem(target_word)
    tokens = []
    # Text Mining Pipeline
    stop_words = set(nltk.corpus.stopwords.words('english')) 
    for s in snippets:
        temp_tokens = nltk.word_tokenize(s)
        filtered_tokens = [w.lower() for w in temp_tokens if not w in stop_words and w.isalpha() and len(w) > 3] 
        tokens.extend(filtered_tokens)
    logger.debug(tokens)
    logger.debug('Total number of tokens: %s', len(tokens))
    # Search for target word
    neighborhood = {} 
    for i in range(0, len(tokens)):
        st = ps.stem(tokens[i])
        if st == stem_target_word:
            neighbors = tokens[i-n: i+n+1]
            logger.debug(neighbors)
            for t in neighbors:
                if t not in neighborhood:
                    neighborhood[t] = 0
                neighborhood[t] += 1
    logger.debug(neighborhood)
    # Convert neighborhood into a list of tuples
    neighborhood = [(k, v) for k, v in neighborhood.items() if v > 1] 
    neighborhood.sort(key=lambda tup: tup[1], reverse=True)
    logger.debug(neighborhood)
    limit = int(len(neighborhood)*0.2)
    
    logger.debug('%s/%s', len(neighborhood), limit)
    neighborhood = neighborhood[:limit]
    
    #x_val = [x[0] for x in neighborhood[:limit]]
    #y_val = [x[1] for x in neighborhood[:limit]]
    #logger.info(len(x_val))
    #plt.plot(x_val,y_val)
    #plt.show()
    return neighborhood



def main(args):
    cws = CWS(config.key)
    cache_ws = CacheSearch(cws, 'cache')

    dpw_cache = {}

    reader = csv.reader(args.d, delimiter=',')
    for row in reader:
        word_a = row[0]
        word_b = row[1]

        word_a_neighborhood = extract_neighborhood(word_a, cache_ws, args.n)
        word_b_neighborhood = extract_neighborhood(word_b, cache_ws, args.n)

        #logger.info('%s (%s)', word_a, len(word_a_neighborhood))
        #logger.info('%s (%s)', word_b, len(word_b_neighborhood))
        dpw_a = DPW(word_a, word_a_neighborhood)
        logger.info(dpw_a)
        dpw_b = DPW(word_b, word_b_neighborhood)
        logger.info(dpw_b)

        nmf_optimization(dpw_a, dpw_cache)
        
        score= dpw_a.similarity(dpw_b)
        #print(f'{word_a},{word_b},{score}')
        print(score)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground')
    parser.add_argument('-d', type=argparse.FileType('r'), required=True, help='dataset file (csv)')
    parser.add_argument('-n', type=int, help='neighborhood size', default=3)
    #parser.add_argument('--r2', type=float, help='R2', )
    #parser.add_argument('-t', type=float, help='Sensitivity', default=1.0)
    #parser.add_argument('-r', type=bool, help='Ranking relative', default=True)
    #parser.add_argument('-m', type=Method, choices=list(Method), default='kneedle')
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)
