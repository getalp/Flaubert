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

from tools.clean_text import cleaner


def get_labels(line):
    """
    Input: line
    Returns pairs of sentences and corresponding labels
    """
    sent1, sent2, label = line.split('\t')
    sent1 = cleaner(sent1, rm_new_lines=True)
    sent2 = cleaner(sent2, rm_new_lines=True)
    sent_pair = '\t'.join([sent1, sent2])
    label = label.strip()

    return sent_pair, label


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--indir', type=str, help='Path to raw data directory')

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
                    sent_pair, label = get_labels(line)
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