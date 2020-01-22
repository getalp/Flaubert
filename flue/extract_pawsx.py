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
    _, sent1, sent2, label = line.split('\t')
    sent1 = cleaner(sent1, rm_new_lines=True, do_lower=do_lower)
    sent2 = cleaner(sent2, rm_new_lines=True, do_lower=do_lower)
    sent_pair = '\t'.join([sent1, sent2])
    label = int(label.strip())

    return sent_pair, label


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--indir', type=str, help='Path to raw data directory')
    parser.add_argument('--outdir', type=str, help='Path to processed data directory')
    parser.add_argument('--do_lower', type=bool_flag, default='False', help='True if do lower case, False otherwise.')
    parser.add_argument('--use_hugging_face', type=bool_flag, default='False', help='Prepare data to run fine-tuning using \
                                                                                    Hugging Face Transformer library')

    args = parser.parse_args()

    indir = os.path.expanduser(args.indir)
    outdir = os.path.expanduser(args.outdir)
    lang = 'fr'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    splts = {'train': ['translated_train.tsv', 'train.tsv', 'train_0.tsv'], 
            'dev': ['dev_2k.tsv', 'dev.tsv', 'valid_0.tsv'], 
            'test': ['test_2k.tsv', 'test.tsv', 'test_0.tsv']}
    idx = 1 if args.use_hugging_face else 2 # index of values in splts

    for s in splts:
        sent_pairs = []
        labels = []
        count = 0
        with open(os.path.join(indir, lang, splts[s][0]), 'rt', encoding='utf-8') as f_in:
            next(f_in)
            with open(os.path.join(outdir, splts[s][idx]), 'w') as f_out:
                tsv_output = csv.writer(f_out, delimiter='\t')
                for i, line in enumerate(f_in):
                    if i == 0 and args.use_hugging_face:
                        tsv_output.writerow(['Label', '', '', 'Sent1', 'Sent2'])
                    sent_pair, label = get_labels(line, do_lower=args.do_lower)
                    if len(sent_pair.split()) > 5:
                        sent_pairs.append(sent_pair)
                        labels.append(label)
                        
                        tsv_output.writerow([label, str(i), str(i), sent_pair])

        assert len(sent_pairs) == len(labels)

        print('Finished processing {}. Positive/Negative: {}/{}'.format(s, sum(labels), len(labels)-sum(labels)))

if __name__ == "__main__":
    main()