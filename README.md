# Flaubert
Unsupervised Language Model Pre-training for French 

**Fully Avaialble soon**

# Table of Contents
**1. [FlauBERT](#FlauBERT)**  
&nbsp;&nbsp;&nbsp;&nbsp;1.1. [FlauBERT-BASE](#FlauBERT-BASE)  
&nbsp;&nbsp;&nbsp;&nbsp;1.2. [FlauBERT-LARGE](#FlauBERT-LARGE)
&nbsp;&nbsp;&nbsp;&nbsp;1.3. [Use FlauBERT with Hugging Face's `transformers`](#Use-FlauBERT-with-Hugging-Face's-`transformers`)
**2. [Training Corpora](#Training-Corpora)**  
**3. [FLUE](#FLUE)**  
&nbsp;&nbsp;&nbsp;&nbsp;3.1. [Text Classification](#Text-Classification)  
&nbsp;&nbsp;&nbsp;&nbsp;3.2. [Paraphrasing](#Paraphrasing)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3. [Natural Language Inference](#Natural-Language-Inference)  
&nbsp;&nbsp;&nbsp;&nbsp;3.4. [Constituency Parsing](#Constituency-Parsing)  
&nbsp;&nbsp;&nbsp;&nbsp;3.5. [Word Sense Disambiguation](#Word-Sense-Disambiguation)  

## FlauBERT

### FlauBERT-BASE

To train FlauBERT-BASE, use the following command

```bash
TBD
```

Pre-trained FlauBERT-BASE compatible with [hugging_face/transformers](https://github.com/huggingface/transformers) is availalbe at [this](https://filesender.renater.fr/?s=download&token=83bedf23-2925-9116-3c7d-10b3e14c5fef) adress.

### FlauBERT-LARGE

To train FlauBERT-LARGE, use the following command

```bash
TBD
```

Pre-trained FlauBERT-LARGE is availalbe **TBD**

## Use FlauBERT with Hugging Face's `transformers`

A Hugging Face's [`transformers`](https://github.com/huggingface/transformers) compatible version of FlauBERT-BASE is available for download [here](https://zenodo.org/record/3562902#.Xef0-i2ZN0s), in an archive named `xlm_bert_fra_base_lower.tar`.

Setup:

```bash
wget https://zenodo.org/record/3562902/files/xlm_bert_fra_base_lower.tar
tar xzf xlm_bert_fra_base_lower.tar
```

Then, you can use the following lines of code:

```python
import torch
from transformers import XLMModel, XLMTokenizer
modelname="xlm_bert_fra_base_lower" # Or absolute path to where you put the folder

# Load model
flaubert, log = XLMModel.from_pretrained(modelname, output_loading_info=True)
# check import was successful, the dictionary should have empty lists as values
print(log)

# Load tokenizer
flaubert_tokenizer = XLMTokenizer.from_pretrained(modelname, do_lower_case=False)

# this line is important: by default, XLMTokenizer removes diacritics, even with do_lower_case=False flag
flaubert_tokenizer.do_lowercase_and_remove_accent = False

sentence="Le chat mange une pomme."
sentence_lower = sentence.lower()

token_ids = torch.tensor([flaubert_tokenizer.encode(sentence_lower)])
last_layer = flaubert(token_ids)[0]
print(last_layer.shape)
#torch.Size([1, 5, 768])  -> (batch size x number of tokens x transformer dimension)
```


## Training Corpora

## FLUE
A general benchmark for evaluating French natural language understanding systems

### Text Classification
The Cross-Lingual Sentiment CLS dataset is publicly available and can be downloaded at [this](https://webis.de/data/webis-cls-10.html) adress.

To fine-tune FlauBERT on the CLS dataset:
```bash
TBD
```

To evaluate FlauBERT on the CLS dataset:
```bash
TBD
```

### Paraphrasing
The Cross-lingual Adversarial dataset for Paraphrase Identification PAWS-X is publicly available and can be downloaded at [this](https://github.com/google-research-datasets/paws) adress.


To fine-tune FlauBERT on the PAWS-X dataset:
```bash
TBD
```

To evaluate FlauBERT on the PAWS-X dataset:
```bash
TBD
```


### Natural Language Inference
The Cross-lingual Natural Language Inference Corpus (XNLI) corpus is publicly available and can be downloaded at [this](https://www.nyu.edu/projects/bowman/xnli/) adress.


To fine-tune FlauBERT on the XNLI corpus:
```bash
TBD
```

To evaluate FlauBERT on the XNLI corpus:
```bash
TBD
```

### Constituency Parsing

The French Treebank collection is freely available for research purposes.
See [here](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php?langue=en) to download the latest version of the corpus and sign the license, and [here](http://dokufarm.phil.hhu.de/spmrl2014/) to obtain the version of the corpus used for the experiments described in the paper.

To fine-tune FlauBERT on constituency parsing on the French Treebank, see instructions [here](https://github.com/mcoavoux/self-attentive-parser/blob/camembert/README_flaubert.md).

Pretrained parsing models will be available soon!


<!---
To fine-tune FlauBERT on the French Treebank collection:
```bash
TBD
```

To evaluate FlauBERT on the French Treebank collection:
```bash
TBD
```
-->

### Word Sense Disambiguation
#### Verb Sense Disambiguation
The FrenchSemEval evaluation dataset is available at [this](http://www.llf.cnrs.fr/dataset/fse/) adress.

To fine-tune FlauBERT on the FrenchSemEval:
```bash
TBD
```

To evaluate FlauBERT on the FrenchSemEval:
```bash
TBD
```

#### Noun Sense Disambiguation
The French Word Sense Disambiguation dataset used in our experiments is publicly available and can be downloaded at [this](https://zenodo.org/record/3549806) adress. 


To fine-tune FlauBERT on **TBD** (need short name for the dataset) :
```bash
TBD
```

To evaluate FlauBERT on **TBD** (need short name for the dataset):
```bash
TBD
```
