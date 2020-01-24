"""
Modified by Hang Le
The original copyright is appended below
--
Copyright (c) 2019-present, Facebook, Inc.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# Code to generate sentence representations from a pretrained model.

import os, sys
import argparse
import torch

import fastBPE

sys.path.append(os.getcwd())
from xlm.model.embedder import SentenceEmbedder
from xlm.data.dictionary import PAD_WORD

class Params:
    def __init__(self, max_batch_size=0):
        self.max_batch_size = max_batch_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Path to the directory where pretrained models are saved.')
    args = parser.parse_args()

    # Get parameters from model_dir
    model_dir = os.path.expanduser(args.model_dir)
    model_dir = model_dir if not model_dir[-1] == '/' else model_dir[:-1]
    model_path = os.path.join(model_dir, [f for f in os.listdir(model_dir) if f.endswith('pth')][0])
    codes_path = os.path.join(model_dir, 'codes')
    vocab_path = os.path.join(model_dir, 'vocab')
    do_lowercase = True if os.path.basename(model_dir).split('_')[-2] == 'uncased' else False

    bpe = fastBPE.fastBPE(codes_path, vocab_path)

    sentences = "Le chat mange une pomme ."
    if do_lowercase:
        sentences = sentences.lower()
    
    # Apply BPE
    sentences = bpe.apply([sentences])
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]
    print(sentences)

    # Create batch
    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    # Reload pretrained model
    params = Params()
    embedder = SentenceEmbedder.reload(model_path, params)
    embedder.eval()
    dico = embedder.dico

    # Prepare inputs to model
    word_ids = torch.LongTensor(slen, bs).fill_(dico.index(PAD_WORD))
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent
    lengths = torch.LongTensor([len(sent) for sent in sentences])

    # Get sentence embeddings
    tensor = embedder.get_embeddings(x=word_ids, lengths=lengths)
    print(tensor.size())

if __name__ == "__main__":
    main()