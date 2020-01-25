"""
Code to generate sentence representations from a pretrained model.
NOTE: This code only works for the modified XLM provided by 
the Flaubert repo: https://github.com/getalp/Flaubert/tree/master/xlm
Copyright (c) 2019-present CNRS and Facebook Inc.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import torch
import fastBPE

# Add Flaubert root to system path (change accordingly)
FLAUBERT_ROOT = '/home/user/Flaubert'
sys.path.append(FLAUBERT_ROOT)

from xlm.model.embedder import SentenceEmbedder
from xlm.data.dictionary import PAD_WORD


# Paths to model files
model_path = '/home/user/flaubert_base_cased/flaubert_base_cased_xlm.pth'
codes_path = '/home/user/flaubert_base_cased/codes'
vocab_path = '/home/user/flaubert_base_cased/vocab'
do_lowercase = False # Change this to True if you use uncased FlauBERT

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
embedder = SentenceEmbedder.reload(model_path)
embedder.eval()
dico = embedder.dico

# Prepare inputs to model
word_ids = torch.LongTensor(slen, bs).fill_(dico.index(PAD_WORD))
for i in range(len(sentences)):
    sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
    word_ids[:len(sent), i] = sent
lengths = torch.LongTensor([len(sent) for sent in sentences])

# Get sentence embeddings (corresponding to the BERT [CLS] token)
cls_embedding = embedder.get_embeddings(x=word_ids, lengths=lengths)
print(cls_embedding.size())

# Get the entire output tensor for all tokens
# Note that cls_embedding = tensor[0]
tensor = embedder.get_embeddings(x=word_ids, lengths=lengths, all_tokens=True)
print(tensor.size())