# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import csv
import enum
import tqdm
import config
import pickle
import logging
import argparse


from cProfile import Profile
from pstats import SortKey, Stats


import semantic.dp as dp
import semantic.corpus as corpus


import exectimeit.timeit as timeit


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


# Enum for size units
class SIZE_UNIT(enum.Enum):
   BYTES = 1
   KB = 2
   MB = 3
   GB = 4


def convert_unit(size_in_bytes, unit=SIZE_UNIT.MB):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if unit == SIZE_UNIT.KB:
       return size_in_bytes/1024
   elif unit == SIZE_UNIT.MB:
       return size_in_bytes/(1024*1024)
   elif unit == SIZE_UNIT.GB:
       return size_in_bytes/(1024*1024*1024)
   else:
       return size_in_bytes


def get_file_size(file_name, size_type=SIZE_UNIT.MB):
   """ Get file in size in given unit like KB, MB or GB"""
   size = os.path.getsize(file_name)
   return convert_unit(size, size_type)


def time_training(terms, corpus_obj, n):
    model = dp.DPWCModel(corpus=corpus_obj, n=n, c=dp.Cutoff.pareto20, latent=True, k=2)
    model.fit(terms)
    return model

@timeit.exectime(1)
def time_inference(model, vocabulary):
    for a in vocabulary:
        for b in vocabulary:
            model.predict(a, b)


def main(args):
    model = None
    
    # Load data model
    if args.l is not None:
        logger.info(f'Load semantic model: {args.l}')
        with open(args.l, 'rb') as input_file:
            model = pickle.load(input_file)

    # Compute inference time
    profiler = Profile()
    profiler.enable()
    ti, std, _ = time_inference(model, model.vocabulary())
    profiler.disable()
    logger.info(f'Inference time {ti}±{std}')
    stats = Stats(profiler).sort_stats('time')
    stats.print_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground - Performance optimization')
    parser.add_argument('-l', type=str, required=True, help='input model')
    args = parser.parse_args()
    main(args)
