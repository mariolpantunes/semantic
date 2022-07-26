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


import semantic.dp as dp
import semantic.corpus as corpus


import exectime.timeit as timeit


logging.basicConfig(level=logging.INFO, format='%(message)s')
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


def time_inference(model, lines):
    for row in csv.reader(lines):
        model.predict(row[0], row[1])


def main(args):
    # Setup corpus
    corpus_obj = corpus.WebCorpus(config.key, config.cache)
    # Get the list of terms
    terms = []
    with open(args.d, 'r') as in_csv:
        #reader = csv.reader(in_csv, delimiter=',')
        lines = [line for line in in_csv]
        for row in csv.reader(lines):
            terms.append(row[0])
            terms.append(row[1])
    terms = list(set(terms))
    logger.info(f'Terms: {terms}')

    # Parameters
    neighborhoods = [3, 5, 7]

    training_time = []
    model_size = []
    inference_time = []

    for n in tqdm.tqdm(neighborhoods):
        # Training time
        ti, std, model = timeit.timeit(1, time_training, terms, corpus_obj, n)
        training_time.append((n,ti,std))
        # Store model and get size
        with open(f'model_{n}.pkl', 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
        model_size.append((n, get_file_size(f'model_{n}.pkl')))
        # Inference time
        ti, std, _ = timeit.timeit(1, time_inference, model, lines)
        inference_time.append((n,ti,std))

    # Print the results
    logger.info(f'Training times ({args.d}/{len(terms)})')
    for n, ti, std in training_time:
        logger.info(f'{n} -> {ti}±{std}')
    
    logger.info(f'Inference times ({args.d}/{len(lines)})')
    for n, ti, std in inference_time:
        logger.info(f'{n} -> {ti}±{std}')
    
    logger.info(f'Model Size ({args.d}/{len(model_size)})')
    for n, s in model_size:
        logger.info(f'{n} -> {s}MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground')
    parser.add_argument('-d', type=str, required=True, help='dataset file (csv)')
    args = parser.parse_args()
    main(args)