"""
Hang Le (hangtp.le@gmail.com)
"""
import os
import numpy as np
import math
import random
import re
import csv
import argparse

import sys
sys.path.append(os.getcwd())

from tools.clean_text import cleaner
from xlm.utils import bool_flag


def get_labels(line, do_lower=False):
    """
    Input: line
    Returns pairs of sentences and corresponding labels
    """
    sent1, sent2, label = line.split('\t')
    sent1 = cleaner(sent1, rm_new_lines=True, do_lower=do_lower)
    sent2 = cleaner(sent2, rm_new_lines=True, do_lower=do_lower)
    sent_pair = '\t'.join([sent1, sent2])
    label = label.strip()

    return sent_pair, label


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--indir', type=str, help='Path to processed data directory')
    parser.add_argument('--do_lower', type=bool_flag, default='False', help='True if do lower case, False otherwise.')

    args = parser.parse_args()

    path = os.path.expanduser(args.indir)

    splts = ['valid', 'test', 'train']
    lang = 'fr'

    for s in splts:
        sent_pairs = []
        labels = []

        with open(os.path.join(path, lang+'.raw.'+s), 'rt', encoding='utf-8') as f_in:
            next(f_in)
            with open(os.path.join(path, '{}_0.xlm.tsv'.format(s)), 'w') as f_out:
                tsv_output = csv.writer(f_out, delimiter='\t')
                for line in f_in:
                    sent_pair, label = get_labels(line, do_lower=args.do_lower)
                    sent_pairs.append(sent_pair)
                    labels.append(label)

                    tsv_output.writerow([sent_pair, label])

        assert len(sent_pairs) == len(labels)

        print('Finished writing {}.review to {}. Neutral/Contradiction/Entailment: {}/{}/{}'.format(s, 
                                                                                                path, 
                                                                                                labels.count('neutral'),
                                                                                                labels.count('contradiction'),
                                                                                                labels.count('entailment')))

if __name__ == "__main__":
    main()