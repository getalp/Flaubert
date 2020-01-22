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


def review_extractor(line, category='dvd', do_lower=False):
    """
    Extract review and label
    """
    m = re.search('(?<=<rating>)\d+.\d+(?=<\/rating>)', line)
    label = 1 if int(float(m.group(0))) > 3 else 0 # rating == 3 are already removed

    if category == 'dvd':
        m = re.search('(?<=\/url><text>)(.|\n|\t|\f)+(?=\<\/title><summary>)', line)
    else:
        m = re.search('(?<=\/url><text>)(.|\n|\t|\f)+(?=\<\/text><title>)', line)

    review_text = m.group(0)

    if do_lower:
        review_text = review_text.lower()

    return review_text, label


def get_review_labels(line, category='dvd', do_lower=False):
    """
    Input: line
    Returns cleaned review and label
    """
    review_text, label = review_extractor(line, category=category, do_lower=do_lower)
    review_text = cleaner(review_text, rm_new_lines=True)

    return review_text, label


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--indir', type=str, help='Path to raw data directory.')
    parser.add_argument('--outdir', type=str, help='Path to processed data directory.')
    parser.add_argument('--do_lower', type=bool_flag, default='False', help='True if do lower case, False otherwise.')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio to split data for validation.')
    parser.add_argument('--use_hugging_face', type=bool_flag, default='False', help='Prepare data to run fine-tuning using \
                                                                                    Hugging Face Transformer library')

    args = parser.parse_args()

    indir = os.path.expanduser(args.indir)
    outdir = os.path.expanduser(args.outdir)

    categories = ['books', 'dvd', 'music']
    lang = 'fr'
    val_ratio = args.val_ratio

    train_fname = 'train.tsv' if args.use_hugging_face else 'train_0.tsv' 
    val_fname = 'dev.tsv' if args.use_hugging_face else 'valid_0.tsv' 
    test_fname = 'test.tsv' if args.use_hugging_face else 'test_0.tsv'  

    for category in categories:
        print('-'*20)
        path = os.path.join(indir, lang, category)
        splts = ['train', 'test']

        for s in splts:
            review_texts = []
            labels = []
            stats = []

            with open(os.path.join(path, s+'.review'), 'rt', encoding='utf-8') as f_in:
                next(f_in)
                i = 0
                text = f_in.read()
                for line in text.split('\n\n'):
                    if len(line) > 9:
                        i += 1
                        review_text, label = get_review_labels(line, category=category, do_lower=args.do_lower)
                        review_texts.append(review_text)
                        labels.append(label)
                        stats.append(len(review_text.split()))
                    else:
                        break
            
            assert len(review_texts) == len(labels) == i

            out_path = os.path.join(outdir, category)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            if s == 'test':
                with open(os.path.join(out_path, test_fname), 'w') as f_out:
                    tsv_output = csv.writer(f_out, delimiter='\t')
                    if args.use_hugging_face:
                        tsv_output.writerow(['Text', 'Label']) # write first line
                    for idx, line in enumerate(review_texts):
                        tsv_output.writerow([line, labels[idx]])

                    print('Finished writing {}.review to {}. Pos/Neg: {}/{}'.format(s, out_path, sum(labels), len(labels)-sum(labels)))
            else:
                # Split train set into train and valid sets
                pos_ids = [i for i, l in enumerate(labels) if l == 1]
                neg_ids = [i for i, l in enumerate(labels) if l == 0]

                num_pos_train = math.ceil(len(pos_ids) * (1 - val_ratio))
                num_neg_train = math.ceil(len(neg_ids) * (1 - val_ratio))

                random.shuffle(pos_ids)
                random.shuffle(neg_ids)

                print('Positive/Negative in original train set of {}: {}/{}'.format(category, len(pos_ids), len(neg_ids)))

                train_ids = pos_ids[:num_pos_train] + neg_ids[:num_neg_train]
                val_ids = pos_ids[num_pos_train:] + neg_ids[num_neg_train:]

                random.shuffle(train_ids)
                random.shuffle(val_ids)
                
                with open(os.path.join(out_path, train_fname), 'w') as f_out:
                    tsv_output = csv.writer(f_out, delimiter='\t')
                    if args.use_hugging_face:
                        tsv_output.writerow(['Text', 'Label']) # write first line
                    for idx in train_ids:
                        tsv_output.writerow([review_texts[idx], labels[idx]])

                with open(os.path.join(out_path, val_fname), 'w') as f_out:
                    tsv_output = csv.writer(f_out, delimiter='\t')
                    if args.use_hugging_face:
                        tsv_output.writerow(['Text', 'Label']) # write first line
                    for idx in val_ids:
                        tsv_output.writerow([review_texts[idx], labels[idx]])

                train_labels = [l for idx, l in enumerate(labels) if idx in train_ids]  
                print('Finished writing {}.review to {}. Pos/Neg: {}/{}'.format('train', out_path, 
                                                                                sum(train_labels), len(train_labels)-sum(train_labels)))
                val_labels = [l for idx, l in enumerate(labels) if idx in val_ids] 
                print('Finished writing {}.review to {}. Pos/Neg: {}/{}'.format('valid', out_path, 
                                                                                sum(val_labels), len(val_labels)-sum(val_labels)))

if __name__ == "__main__":
    main()