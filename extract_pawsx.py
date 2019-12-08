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
    _, sent1, sent2, label = line.split('\t')
    sent1 = cleaner(sent1, rm_new_lines=True)
    sent2 = cleaner(sent2, rm_new_lines=True)
    sent_pair = '\t'.join([sent1, sent2])
    label = int(label.strip())

    return sent_pair, label


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--indir', type=str, help='Path to raw data directory')
    parser.add_argument('--outdir', type=str, help='Path to processed data directory')

    args = parser.parse_args()

    path_raw = os.path.expanduser(args.indir)
    path_proc = os.path.expanduser(args.outdir)
    lang = 'fr'

    if not os.path.exists(path_proc):
        os.makedirs(path_proc)

    splts = {'valid': 'dev_2k', 'test': 'test_2k', 'train': 'translated_train'}

    for s in splts:
        sent_pairs = []
        labels = []
        count = 0
        with open(os.path.join(path_raw, lang, splts[s]+'.tsv'), 'rt', encoding='utf-8') as f_in:
            next(f_in)
            if s == 'valid':
                s = 'dev'
            with open(os.path.join(path_proc, '{}_0.tsv'.format(s)), 'w') as f_out:
                tsv_output = csv.writer(f_out, delimiter='\t')
                for i, line in enumerate(f_in):
                    if i == 0:
                        tsv_output.writerow(['label', '', '', 'Sent1', 'Sent2'])

                    sent_pair, label = get_labels(line)
                    if len(sent_pair.split()) > 5:
                        sent_pairs.append(sent_pair)
                        labels.append(label)
                        
                        tsv_output.writerow([label, str(i), str(i), sent_pair])

        assert len(sent_pairs) == len(labels)

        print('Finished processing split {}. Positive/Negative: {}/{}'.format(s, sum(labels), len(labels)-sum(labels)))

if __name__ == "__main__":
    main()