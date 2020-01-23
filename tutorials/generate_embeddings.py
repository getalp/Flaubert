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

BPE_CODES_VOCAB = {'base_uncased': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_uncased_xlm'),
        'base_cased': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_cased_xlm'),
        'large_cased_v0': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_large_cased_xlm_v0'),
        'large_cased': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_large_cased_xlm_v0')}

MODELPATHS_XLM = {'base_uncased': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_uncased_xlm/best-valid_fr_mlm_ppl.pth'),
        'base_cased': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_cased_xlm/best-valid_fr_mlm_ppl.pth'),
        'large_cased_v0': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_large_cased_xlm_v0/best-valid_fr_mlm_ppl.pth'),
        'large_cased': os.path.expanduser('~/pretrained_models/Flaubert/flaubert_large_cased_xlm/best-valid_fr_mlm_ppl.pth')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='base_cased', 
                        choices=['base_uncased', 'base_cased', 'large_cased_v0', 'large_cased'])
    args = parser.parse_args()

    model_type = args.model_type
    model_path = MODELPATHS_XLM[model_type]
    do_lowercase = True if model_type.split('_')[1] == 'uncased' else False

    codes_path = os.path.join(BPE_CODES_VOCAB[model_type], 'codes')
    vocab_path = os.path.join(BPE_CODES_VOCAB[model_type], 'vocab')
    bpe = fastBPE.fastBPE(codes_path, vocab_path)

    sentences = "Le chat mange une pomme ."
    if do_lowercase:
        sentences = sentences.lower()
    
    sentences = bpe.apply([sentences])
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]
    print(sentences)

    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    class Params:
        def __init__(self, max_batch_size=0):
            self.max_batch_size = max_batch_size

    params = Params()
    embedder = SentenceEmbedder.reload(model_path, params)
    embedder.eval()
    dico = embedder.dico

    word_ids = torch.LongTensor(slen, bs).fill_(dico.index(PAD_WORD))
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])
    print('word_ids: {}'.format(word_ids))
    print('lengths: {}'.format(lengths))

    tensor = embedder.get_embeddings(x=word_ids, lengths=lengths)
    print(tensor.size())
    print(torch.norm(tensor))

if __name__ == "__main__":
    main()