"""
Hang Le (hangtp.le@gmail.com)
"""
import sys
import numpy as np
import argparse
import nltk
from nltk.tokenize import sent_tokenize


def split_sentences(text):
    """
    Split text into 1 sentence per line
    """
    # Tokenize sentences
    sent_list = sent_tokenize(text)
    
    return '\n'.join(sent_list)


for line in sys.stdin:
    line = split_sentences(line)
    print(u'%s' % line)