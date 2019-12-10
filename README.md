# Flaubert
Unsupervised Language Model Pre-training for French 

**Fully Avaialble soon**

# Table of Contents
**1. [FlauBERT](#FlauBERT)**  
&nbsp;&nbsp;&nbsp;&nbsp;1.1. [FlauBERT-BASE](#FlauBERT-BASE)  
&nbsp;&nbsp;&nbsp;&nbsp;1.2. [FlauBERT-LARGE](#FlauBERT-LARGE)  
&nbsp;&nbsp;&nbsp;&nbsp;1.3. [Use FlauBERT with Hugging Face's `transformers`](#use-flaubert-with-hugging-faces-transformers)  
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

### FlauBERT-LARGE

To train FlauBERT-LARGE, use the following command

```bash
TBD
```

Pre-trained FlauBERT-LARGE is availalbe **TBD**

### Use FlauBERT with Hugging Face's `transformers`

A Hugging Face's [`transformers`](https://github.com/huggingface/transformers) compatible version of FlauBERT-BASE is available for download [here](https://zenodo.org/record/3567594#.Xe4Zmi2ZN0t), in an archive named `xlm_bert_fra_base_lower.tar`.

Setup:

```bash
wget https://zenodo.org/record/3567594/files/xlm_bert_fra_base_lower.tar
tar xf xlm_bert_fra_base_lower.tar
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
Coming soon.

## FLUE
A general benchmark for evaluating French natural language understanding systems

#### Install dependencies
To preprocess data we need to clone [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses tokenizer](https://github.com/moses-smt/mosesdecoder):
```bash
cd tools
git clone https://github.com/glample/fastBPE.git
git clone https://github.com/moses-smt/mosesdecoder.git
```

### Text Classification
In the following, you should replace `$DATA_DIR` with a location on your computer, e.g. `~/data/cls`, `~/data/pawsx`, `~/data/xnli`, etc. depending on the task.

#### Download data
<!-- The Cross-Lingual Sentiment CLS dataset is publicly available and can be downloaded at [this](https://webis.de/data/webis-cls-10.html) adress. -->
Excecute the following command:
```
bash get-data-cls.sh $DATA_DIR
```

#### Preprocess data
<!-- The following command:
```
python extract_split_cls.py --indir $DATA_DIR/raw/cls-acl10-unprocessed -outdir $DATA_DIR/processed
``` 
will split the training set into training and validation sets and save them to `$DATA_DIR/processed/[category]/[split].tsv`, where:
- `[category]` is either `books` or `dvd` or `music`.
- `[split]` is either `train` or `dev` or `test`. -->

Run the following command:
```bash
bash prepare-data-cls.sh $DATA_DIR
```

#### Finetune on the CLS dataset
To fine-tune and evaluate FlauBERT on the CLS dataset, we need to first install [Hugging Face's Transformers](https://github.com/huggingface/transformers) from their repo:
```
pip install git+https://github.com/huggingface/transformers.git --upgrade
```

To fine-tune, we use the finetuning script for GLUE benchmark from [Hugging Face's Transformers](https://github.com/huggingface/transformers):
```
python run_flue.py \
  --model_type xlm \
  --model_name_or_path xlm_bert_fra_base_lower \
  --task_name SST-2 \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR/processed/books
  --max_seq_length 512 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 30 \
  --output_dir ./dumped \
  --overwrite_output_dir \
  --save_steps 10000 \
  |& tee output.log
```
Replace `books` with the category you want (among `books, dvd, music`).
<!-- To evaluate FlauBERT on the CLS dataset:
```bash
TBD
``` -->

### Paraphrasing
<!-- The Cross-lingual Adversarial dataset for Paraphrase Identification PAWS-X is publicly available and can be downloaded at [this](https://github.com/google-research-datasets/paws) adress. -->

#### Download data
```bash
bash get-data-pawsx.sh $DATA_DIR
```

#### Preprocess data
```bash
bash prepare-data-pawsx.sh $DATA_DIR
```

#### Finetune on the PAWS-X dataset
To fine-tune and evaluate FlauBERT on the PAWS-X dataset:
```
python run_flue.py \
  --model_type xlm \
  --model_name_or_path xlm_bert_fra_base_lower \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR/processed \ 
  --max_seq_length 512 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 30 \
  --output_dir ./dumped \
  --overwrite_output_dir \
  --save_steps 10000 \
  |& tee output.log
```

<!-- To evaluate FlauBERT on the PAWS-X dataset:
```bash
TBD
``` -->

### Natural Language Inference
To fine-tune FlauBERT on the XNLI corpus, first download and extract `flaubert_base_lower.zip` from [here](https://zenodo.org/record/3567594#.Xe4Zmi2ZN0t). This file contains:
- `flaubert_base_lower.pth`: FlauBERT's pretrained checkpoint.
- `codes`: BPE codes learned on the training data.
- `vocab`: Vocabulary file.

<!-- The Cross-lingual Natural Language Inference Corpus (XNLI) corpus is publicly available and can be downloaded at [this](https://www.nyu.edu/projects/bowman/xnli/) adress. -->
In the following, `$MODEL_DIR` denotes the above extracted folder.

#### Downnload data
```bash
bash get-data-xnli.sh $DATA_DIR
```

#### Preprocess data
```bash
bash prepare-data-xnli.sh $DATA_DIR $MODEL_DIR
```

#### Finetune on the XNLI corpus

```bash
python flue_xnli.py \
    --exp_name flaubert_base_lower_xnli \
    --dump_path ./dumped  \
    --model_path $MODEL_DIR/flaubert_base_lower.pth \
    --data_path $DATA_DIR/processed  \
    --dropout 0.1 \
    --transfer_tasks FR-XNLI\
    --optimizer_e adam,lr=0.000005 \
    --optimizer_p adam,lr=0.000005 \
    --finetune_layers "0:_1" \
    --batch_size 8 \
    --n_epochs 30 \
    --epoch_size -1 \
    --max_len 512
``` 
<!-- To evaluate FlauBERT on the XNLI corpus:
```bash
TBD
``` -->

### Constituency Parsing

The French Treebank collection is freely available for research purposes.
See [here](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php?langue=en) to download the latest version of the corpus and sign the license, and [here](http://dokufarm.phil.hhu.de/spmrl2014/) to obtain the version of the corpus used for the experiments described in the paper.

To fine-tune FlauBERT on constituency parsing on the French Treebank, see instructions [here](https://github.com/mcoavoux/self-attentive-parser).

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

**Code coming soon**

#### Noun Sense Disambiguation
The French Word Sense Disambiguation dataset used in our experiments is publicly available and can be downloaded at [this](https://zenodo.org/record/3549806) adress. 

**Code coming soon**

