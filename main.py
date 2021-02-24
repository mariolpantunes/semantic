# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import argparse
import logging
import numpy as np


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic playground')
    #parser.add_argument('-i', type=str, required=True, help='input file')
    #parser.add_argument('--r2', type=float, help='R2', default=0.95)
    #parser.add_argument('-t', type=float, help='Sensitivity', default=1.0)
    #parser.add_argument('-r', type=bool, help='Ranking relative', default=True)
    #parser.add_argument('-m', type=Method, choices=list(Method), default='kneedle')
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)
