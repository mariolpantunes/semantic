# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import csv
import tqdm
import config
import logging
import argparse


import semantic.dp as dp
import semantic.corpus as corpus


import exectime.timeit as timeit


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def time_training(terms, corpus_obj, n):
    model = dp.DPWModel(corpus=corpus_obj, n=n, c=dp.Cutoff.pareto20, latent=True, k=2)
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
    inference_time = []

    for n in tqdm.tqdm(neighborhoods):
        # Training time
        ti, std, model = timeit.timeit(3, time_training, terms, corpus_obj, n)
        training_time.append((n,ti,std))
        # Inference time
        ti, std, _ = timeit.timeit(3, time_inference, model, lines)
        training_time.append((n,ti,std))


    # Print the results
    logger.info(f'Training times ({args.d})')
    for n, ti, std in training_time:
        logger.info(f'{n} -> {ti}±{std}')
    logger.info(f'Inference times ({args.d})')
    for n, ti, std in inference_time:
        logger.info(f'{n} -> {ti}±{std}')
    logger.info(f'Model Size ({args.d})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground')
    parser.add_argument('-d', type=str, required=True, help='dataset file (csv)')
    args = parser.parse_args()
    main(args)