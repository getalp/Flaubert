# FlauBERT and FLUE

**FlauBERT** is a French BERT trained on a very large and heterogeneous French corpus. Models of different sizes are trained using the new CNRS  (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/eng/jean-zay/ ) supercomputer. This repository shares everything: pre-trained models (base and large), the data, the code to use the models and the code to train them if you need. 
 
Along with FlauBERT comes [**FLUE**](https://github.com/getalp/Flaubert/tree/master/flue): an evaluation setup for French NLP systems similar to the popular GLUE benchmark. The goal is to enable further reproducible experiments in the future and to share models and progress on the French language. 
 
This repository is **still under construction** and everything will be available soon. 


# Table of Contents
**1. [Using FlauBERT](#1.-Using-FlauBERT)**   
&nbsp;&nbsp;&nbsp;&nbsp;1.1. [Using FlauBERT with Hugging Face's `transformers`](#1.1.-Using-FlauBERT-with-Hugging-Face's-`transformers`)   
&nbsp;&nbsp;&nbsp;&nbsp;1.2. [Using FlauBERT with XLM's repository](#1.2.-Using-FlauBERT-with-Facebook-XLM's-repository)  
**2. [Pretraining FlauBERT](#2.-Pretraining-FlauBERT)**  
&nbsp;&nbsp;&nbsp;&nbsp;2.1. [Data](#2.1.-Data)  
&nbsp;&nbsp;&nbsp;&nbsp;2.2. [Training](#2.2.-Training)  
**3. [Fine-tuning FlauBERT on the FLUE benchmark](#3.-Fine-tuning-FlauBERT-on-the-FLUE-benchmark)**  
**4. [Citation](#4.-Citation)** 
<!-- &nbsp;&nbsp;&nbsp;&nbsp;3.1. [Text Classification](#Text-Classification)  
&nbsp;&nbsp;&nbsp;&nbsp;3.2. [Paraphrasing](#Paraphrasing)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3. [Natural Language Inference](#Natural-Language-Inference)  
&nbsp;&nbsp;&nbsp;&nbsp;3.4. [Constituency Parsing](#Constituency-Parsing)  
&nbsp;&nbsp;&nbsp;&nbsp;3.5. [Word Sense Disambiguation](#Word-Sense-Disambiguation)   -->
 

# 1. Using FlauBERT
In this section, we describe two ways to obtain sentence embeddings from pretrained FlauBERT models: either via [Hugging Face's `transformer`](https://github.com/huggingface/transformers) library or via [XLM's library](https://github.com/facebookresearch/XLM). 

## 1.1. Using FlauBERT with Hugging Face's `transformers`
First, you need to install a `transformers` version that contains FlauBERT. At the time of writing this has not been integrated into the official Hugging Face’s repo yet so you would need to install it from our fork:

```
pip install --upgrade --force-reinstall git+https://github.com/formiel/transformers.git
```
(We will make sure to keep this fork up-to-date with the original `transformers` master branch.)

After the installation you can use FlauBERT in a native way:

```bash
import torch
from transformers import FlaubertModel, FlaubertTokenizer

# Choose among ['flaubert-base-cased', 'flaubert-base-uncased', 'flaubert-large-cased']
modelname = 'flaubert-base-cased' 

# Load pretrained model and tokenizer
flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
# do_lowercase=False if using cased models, True if using uncased ones

sentence = "Le chat mange une pomme."
token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])

last_layer = flaubert(token_ids)[0]
print(last_layer.shape)
# torch.Size([1, 8, 768])  -> (batch size x number of tokens x embedding dimension)

# Sentence embeddings is the first hidden state of the last layer (corresponds to the token [CLS] in BERT)
sentence_embedding = last_layer.squeeze()[0]
```

<!-- A Hugging Face's [`transformers`](https://github.com/huggingface/transformers) compatible version of FlauBERT-BASE is available for download [here](https://zenodo.org/record/3567594#.Xe4Zmi2ZN0t), in an archive named `xlm_bert_fra_base_lower.tar`.

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
flaubert_tokenizer = XLMTokenizer.from_pretrained(modelname, do_lowercase_and_remove_accent=False)

sentence="Le chat mange une pomme."
sentence_lower = sentence.lower()

token_ids = torch.tensor([flaubert_tokenizer.encode(sentence_lower)])
last_layer = flaubert(token_ids)[0]
print(last_layer.shape)
#torch.Size([1, 5, 768])  -> (batch size x number of tokens x transformer dimension)
``` -->

 ## 1.2. Using FlauBERT with XLM's repository
The pretrained FlauBERT models are available for downloading in [here](https://zenodo.org/record/3622251). Each compressed folder includes 3 files:
- `flaubert_base_uncased_xlm.pth` (or `flaubert_base_cased_xlm.pth`, `flaubert_large_cased_xlm.pth`): FlauBERT's pretrained model.
- `codes`: BPE codes learned on the training data.
- `vocab`: BPE vocabulary file.

You can obtain sentence embeddings by following [this tutorial](https://github.com/facebookresearch/XLM/blob/master/generate-embeddings.ipynb) in original XLM [repo](https://github.com/facebookresearch/XLM) or refer to our example [here]().

# 2. Pretraining FlauBERT

## 2.1. Data

#### Dependencies
You should clone this repo and then install [WikiExtractor](https://github.com/attardi/wikiextractor), [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses tokenizer](https://github.com/moses-smt/mosesdecoder):
```bash
git clone https://github.com/getalp/Flaubert.git

# Install toolkit
cd tools
git clone https://github.com/attardi/wikiextractor.git
git clone https://github.com/moses-smt/mosesdecoder.git

git clone https://github.com/glample/fastBPE.git
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```

#### Data download and preprocessing
In the following, replace `$DATA_DIR`, `$corpus_name` respectively with the path to the local directory to save the downloaded data and the name of the corpus that you want to download among the options specified in the scripts.

To download and preprocess the data, excecute the following commands:
```bash
./download.sh $DATA_DIR $corpus_name fr
./preprocess.sh $DATA_DIR $corpus_name fr
```

For example:
```bash
./download.sh ~/data gutenberg fr
./preprocess.sh ~/data gutenberg fr
```

The first command will download the raw data to `$DATA_DIR/raw/fr_gutenberg`, the second one processes them and save to `$DATA_DIR/processed/fr_gutenberg`.

For most of the corpora you can also replace `fr` by another language (we may provide a more detailed documentation on this later).

## 2.2. Training
Our codebase for pretraining FlauBERT is largely based on the [XLM repo](https://github.com/facebookresearch/XLM#i-monolingual-language-model-pretraining-bert), with some modifications. You can use their code to train FlauBERT, it will work just fine.

Execute the following command to train FlauBERT (base) on your preprocessed data:

```bash
python train.py \
    --exp_name flaubert_base_lower \
    --dump_path path/to/save/model \
    --data_path path/to/data \
    --lgs 'fr' \
    --clm_steps '' \
    --mlm_steps 'fr' \
    --emb_dim 768 \
    --n_layers 12 \
    --n_heads 12 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --batch_size 16 \
    --bptt 512 \
    --optimizer "adam_inverse_sqrt,lr=0.0006,warmup_updates=24000,beta1=0.9,beta2=0.98,weight_decay=0.01,eps=0.000001" \
    --epoch_size 300000 \
    --max_epoch 100000 \
    --validation_metrics _valid_fr_mlm_ppl \
    --stopping_criterion _valid_fr_mlm_ppl,20 \
    --fp16 true \
    --accumulate_gradients 16 \
    --word_mask_keep_rand '0.8,0.1,0.1' \
    --word_pred '0.15'                      
```

# 3. Fine-tuning FlauBERT on the FLUE benchmark
[FLUE](https://github.com/getalp/Flaubert/tree/master/flue) (French Language Understanding Evaludation) is a general benchmark for evaluating French NLP systems. Please refer to [this page](https://github.com/getalp/Flaubert/tree/master/flue) for an example of fine-tuning FlauBERT on this benchmark.

<!-- ### Text Classification
In the following, you should replace `$DATA_DIR` with a location on your computer, e.g. `~/data/cls`, `~/data/pawsx`, `~/data/xnli`, etc. depending on the task.

#### Download data
Excecute the following command:
```
bash flue/get-data-cls.sh $DATA_DIR
```

#### Preprocess data
Run the following command:
```bash
bash flue/prepare-data-cls.sh $DATA_DIR
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

### Paraphrasing

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

### Natural Language Inference
To fine-tune FlauBERT on the XNLI corpus, first download and extract `flaubert_base_lower.zip` from [here](https://zenodo.org/record/3567594#.Xe4Zmi2ZN0t). This file contains:
- `flaubert_base_lower.pth`: FlauBERT's pretrained checkpoint.
- `codes`: BPE codes learned on the training data.
- `vocab`: Vocabulary file.

<!-- The Cross-lingual Natural Language Inference Corpus (XNLI) corpus is publicly available and can be downloaded at [this](https://www.nyu.edu/projects/bowman/xnli/) adress. -->
<!-- In the following, `$MODEL_DIR` denotes the above extracted folder.

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

<!-- ### Constituency Parsing

The French Treebank collection is freely available for research purposes.
See [here](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php?langue=en) to download the latest version of the corpus and sign the license, and [here](http://dokufarm.phil.hhu.de/spmrl2014/) to obtain the version of the corpus used for the experiments described in the paper.

To fine-tune FlauBERT on constituency parsing on the French Treebank, see instructions [here](https://github.com/mcoavoux/self-attentive-parser). -->

<!-- Pretrained parsing models will be available soon! -->


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

<!-- ### Word Sense Disambiguation
#### Verb Sense Disambiguation
The FrenchSemEval evaluation dataset is available at [this](http://www.llf.cnrs.fr/dataset/fse/) address.

**Code coming soon**

#### Noun Sense Disambiguation

To fine-tune Flaubert for French WSD with WordNet as sense inventory, you can follow the scripts located in the directory `flue/wsd/nouns`, which allow you to:
- Automatically download our publicly available dataset from [this address](https://zenodo.org/record/3549806)  
  → See the script [0.get_data.sh](flue/wsd/nouns/0.get_data.sh)
- Download the `disambiguate` toolkit from [this repository](https://github.com/getalp/disambiguate)  
  → See the script [1.get_toolkit.sh](flue/wsd/nouns/1.get_toolkit.sh)
- Prepare the training/development data from the French SemCor and French WordNet Gloss Corpus  
 → See the script [2.prepare_data.sh](flue/wsd/nouns/2.prepare_data.sh)
- Train the neural model  (assumes that `$FLAUBERT_PATH` is the path to a Flaubert model)  
  → See the script [3.train_model.sh](flue/wsd/nouns/3.train_model.sh)
- Evaluate the model on the French SemEval 2013 task 12 corpus  
  → See the script [4.evaluate_model.sh](flue/wsd/nouns/4.evaluate_model.sh)
  
Once the model is trained, you can disambiguate any text using the script [5.disambiguate.sh](flue/wsd/nouns/5.disambiguate.sh) -->


# 4. Citation
If you use FlauBERT or the FLUE Benchmark for your scientific publication, or if you found the resources in this repository useful, please refer to our [paper](https://arxiv.org/abs/1912.05372):

```
@misc{le2019flaubert,
    title={FlauBERT: Unsupervised Language Model Pre-training for French},
    author={Hang Le and Loïc Vial and Jibril Frej and Vincent Segonne and Maximin Coavoux and Benjamin Lecouteux and Alexandre Allauzen and Benoît Crabbé and Laurent Besacier and Didier Schwab},
    year={2019},
    eprint={1912.05372},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
