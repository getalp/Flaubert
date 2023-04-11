# FlauBERT and FLUE

**FlauBERT** is a French BERT trained on a very large and heterogeneous French corpus. Models of different sizes are trained using the new CNRS  (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/eng/jean-zay/ ) supercomputer. This repository shares everything: pre-trained models (base and large), the data, the code to use the models and the code to train them if you need. 
 
Along with FlauBERT comes [**FLUE**](https://github.com/getalp/Flaubert/tree/master/flue): an evaluation setup for French NLP systems similar to the popular GLUE benchmark. The goal is to enable further reproducible experiments in the future and to share models and progress on the French language. 
 
This repository is **still under construction** and everything will be available soon. 


# Table of Contents
**1. [FlauBERT models](#1-flaubert-models)**  
**2. [Using FlauBERT](#2-using-flaubert)**   
&nbsp;&nbsp;&nbsp;&nbsp;2.1. [Using FlauBERT with Hugging Face's Transformers](#21-using-flaubert-with-hugging-faces-transformers)   
&nbsp;&nbsp;&nbsp;&nbsp;2.2. [Using FlauBERT with Facebook XLM's library](#22-using-flaubert-with-facebook-xlms-library)  
**3. [Pre-training FlauBERT](#3-pre-training-flaubert)**  
&nbsp;&nbsp;&nbsp;&nbsp;3.1. [Data](#31-data)  
&nbsp;&nbsp;&nbsp;&nbsp;3.2. [Training](#32-training)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3. [Convert an XLM pre-trained model to Hugging Face's Transformers](#33-convert-an-XLM-pre-trained-model-to-hugging-faces-transformers)  
**4. [Fine-tuning FlauBERT on the FLUE benchmark](#4-fine-tuning-flaubert-on-the-flue-benchmark)**  
**5. [Citation](#5-citation)** 
<!-- &nbsp;&nbsp;&nbsp;&nbsp;3.1. [Text Classification](#Text-Classification)  
&nbsp;&nbsp;&nbsp;&nbsp;3.2. [Paraphrasing](#Paraphrasing)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3. [Natural Language Inference](#Natural-Language-Inference)  
&nbsp;&nbsp;&nbsp;&nbsp;3.4. [Constituency Parsing](#Constituency-Parsing)  
&nbsp;&nbsp;&nbsp;&nbsp;3.5. [Word Sense Disambiguation](#Word-Sense-Disambiguation)   -->
 
# 1. FlauBERT models
**FlauBERT** is a French BERT trained on a very large and heterogeneous French corpus. Models of different sizes are trained using the new CNRS  (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/eng/jean-zay/ ) supercomputer. We have released the pretrained weights for the following model sizes.

The pretrained models are available for download from [here](https://zenodo.org/record/3627732) or via Hugging Face's library.

| Model name | Number of layers | Attention Heads | Embedding Dimension | Total Parameters |
| :------:       |   :---: | :---: | :---: | :---: |
| `flaubert-small-cased` | 6    | 8    | 512   | 54 M |
| `flaubert-base-uncased`  | 12  | 12  | 768  | 137 M |
| `flaubert-base-cased`   | 12   | 12      | 768   | 138 M |
| `flaubert-large-cased`  | 24   | 16     | 1024 | 373 M |

Note: `flaubert-small-cased` is partially trained so performance is not guaranteed. Consider using it for debugging purpose only.

We also provide the checkpoints from [here](https://www.dropbox.com/s/65f8unz1imz89ew/flaubert_checkpoints.tar.gz?dl=0) for model base (cased/uncased) and large (cased).

# 2. Using FlauBERT
In this section, we describe two ways to obtain sentence embeddings from pretrained FlauBERT models: either via [Hugging Face's Transformer](https://github.com/huggingface/transformers) library or via [Facebook's XLM library](https://github.com/facebookresearch/XLM). We will intergrate FlauBERT into [Facebook' fairseq](https://github.com/pytorch/fairseq) in the near future.

## 2.1. Using FlauBERT with Hugging Face's Transformers
You can use FlauBERT with [Hugging Face's Transformers](https://github.com/huggingface/transformers) library as follow.
<!-- First, you need to install a Transformers version that contains FlauBERT. At the time of writing, our pull request has not been merged into the official Hugging Face’s repo yet so you would need to install it from our fork:

```
pip install --upgrade --force-reinstall git+https://github.com/formiel/transformers.git
```
(We will make sure to keep this fork up-to-date with the original `transformers` master branch.)

After the installation you can use FlauBERT in a native way: -->

```python
import torch
from transformers import FlaubertModel, FlaubertTokenizer

# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased', 
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_base_cased' 

# Load pretrained model and tokenizer
flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
# do_lowercase=False if using cased models, True if using uncased ones

sentence = "Le chat mange une pomme."
token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])

last_layer = flaubert(token_ids)[0]
print(last_layer.shape)
# torch.Size([1, 8, 768])  -> (batch size x number of tokens x embedding dimension)

# The BERT [CLS] token correspond to the first hidden state of the last layer
cls_embedding = last_layer[:, 0, :]
```

**Notes:** if your `transformers` version is <=2.10.0, `modelname` should take one
of the following values:

```
['flaubert-small-cased', 'flaubert-base-uncased', 'flaubert-base-cased', 'flaubert-large-cased']
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

## 2.2. Using FlauBERT with Facebook XLM's library
The pretrained FlauBERT models are available for download from [here](https://zenodo.org/record/3627732). Each compressed folder includes 3 files:
- `*.pth`: FlauBERT's pretrained model.
- `codes`: BPE codes learned on the training data.
- `vocab`: BPE vocabulary file.

**Note:** The following example only works for the modified XLM provided in this repo, it won't work for the [original XLM](https://github.com/facebookresearch/XLM). The code is taken from [this tutorial](https://github.com/getalp/Flaubert/blob/master/tutorials/generate_embeddings.py).

```python
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
```

# 3. Pre-training FlauBERT

### Install dependencies
You should clone this repo and then install [WikiExtractor](https://github.com/attardi/wikiextractor), [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses tokenizer](https://github.com/moses-smt/mosesdecoder) under `tools`:
```bash
git clone https://github.com/getalp/Flaubert.git
cd Flaubert

# Install toolkit
cd tools
git clone https://github.com/attardi/wikiextractor.git
git clone https://github.com/moses-smt/mosesdecoder.git

git clone https://github.com/glample/fastBPE.git
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```

## 3.1. Data
In this section, we describe the pipeline to prepare the data for training FlauBERT. This is based on [Facebook XLM's library](https://github.com/facebookresearch/XLM). The steps are as follows:
1. Download, clean, and tokenize data using Moses tokenizer.
2. Split cleaned data into: train, validation, and test sets.
3. Learn BPE on the training set. Then apply learned BPE codes to train, validation, and test sets.
4. Binarize data.

### (1) Download and Preprocess Data
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

<!-- Below is the list of copora that we use for pre-training FlauBERT along with their corresponding `$corpus_name`s. For most of the corpora you can also replace `fr` by another language if that language is provided. You can refer to our [paper](https://arxiv.org/abs/1912.05372) for more details on the statistics of these corpora (for French only).

| Dataset | `$corpus_name` | Available languages | Note |
| :------|   :--- | :---: | :---: | 
| CommonCrawl |  `common_crawl`   |     |     |
| NewsCrawl |   `news_crawl`  |     |     |
| Wikipedia |   `wiki`  |     |     |
| Wikisource |   `wikisource`  |     |     |
| EU Bookshop |  `feu_bookshop`   |     |     |
| MultiUN |   `multi_un`  |     |     |
| GIGA |  `giga`   |     |     |
| PCT |  NA   |     |     |
| Project Gutenberg |  `gutenberg`   |     |     |
| OpenSubtitles |  `open_subtitles16`   |     |     |
| Le Monde |  NA   |     |     |
| DGT |  `dgt`   |     |     |
| EuroParl |  `europarl`   |     |     |
| EnronSent |  NA   |     |     |
| NewsCommentary |  `news_commentary`   |     |     |
| Wiktionary |  NA   |     |     |
| Global Voices |  `global_voices`   |     |     |
| Wikinews |  `wikinews`   |     |     |
| TED Talks |  ``   |     |     |
| Wikiversity |     |     |     |
| Wikibooks |     |     |     |
| Wikiquote |     |     |     |
| Wikivoyage |     |     |     |
| EUconst |     |     |     | -->

### (2) Split Data
Run the following command to split cleaned corpus into train, validation, and test sets. You can modify the train/validation/test ratio in the script.

```bash
bash tools/split_train_val_test.sh $DATA_PATH
```
where `$DATA_PATH` is path to the file to be split. 

The output files are: `fr.train, fr.valid, fr.test` which are saved under the same directory as the original file.

### (3) & (4) Learn BPE and Prepare Data
Run the following command to learn BPE codes on the training set, and apply BPE codes on the train, validation, and test sets. The data is then binarized and ready for training.
```bash
bash tools/create_pretraining_data.sh $DATA_DIR $BPE_size
```
where `$DATA_DIR` is path to the directory where the 3 above files `fr.train, fr.valid, fr.test` are saved. `$BPE_size` is the number of BPE vocabulary size, for example: `30` for 30k,`50` for 50k, etc. The output files are saved in `$DATA_DIR/BPE/30k` or `$DATA_DIR/BPE/50k` correspondingly.

## 3.2. Training
Our codebase for pretraining FlauBERT is largely based on the [XLM repo](https://github.com/facebookresearch/XLM#i-monolingual-language-model-pretraining-bert), with some modifications. You can use their code to train FlauBERT, it will work just fine.

Execute the following command to train FlauBERT (base) on your preprocessed data:

```bash
python train.py \
    --exp_name flaubert_base_cased \
    --dump_path $dump_path \
    --data_path $data_path \
    --amp 1 \
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
where `$dump_path` is the path to where you want to save your pretrained model, `$data_path` is the path to the binarized data sets, for example `$DATA_DIR/BPE/50k`.

### Run experiments on multiple GPUs and/or multiple nodes
To run experiments on multiple GPUs in a single machine, you can use the following command (the parameters after `train.py` are the same as above).
```bash
export NGPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 # if you only use some of the GPUs in the machine
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

To run experiments on multiple nodes, multiple GPUs in clusters using SLURM as a resource manager, you can use the following command to launch training after requesting resources with `#SBATCH` (the parameters after `train.py` are the same as above plus `--master_port` parameter).
```bash
srun python train.py
```

## 3.3. Convert an XLM pre-trained model to Hugging Face's Transformers
To convert an XLM pre-trained model to Hugging Face's Transformers, you can use the following command.
```bash
python tools/use_flaubert_with_transformers/convert_to_transformers.py --inputdir $inputdir --outputdir $outputdir
```
where `$inputdir` is path to the XLM pretrained model directory, `$outputdir` is path to the output directory where you want to save the Hugging Face's Transformer model.

# 4. Fine-tuning FlauBERT on the FLUE benchmark
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

# 5. Video presentation

You can watch this 7mn video presentation of FlauBERT
[VIDEO 7mn] (https://www.youtube.com/watch?v=NgLM9GuwSwc)

# 6. Citation
If you use FlauBERT or the FLUE Benchmark for your scientific publication, or if you find the resources in this repository useful, please cite one of the following papers:

[LREC paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.302.pdf)
```
@InProceedings{le2020flaubert,
  author    = {Le, Hang  and  Vial, Lo\"{i}c  and  Frej, Jibril  and  Segonne, Vincent  and  Coavoux, Maximin  and  Lecouteux, Benjamin  and  Allauzen, Alexandre  and  Crabb\'{e}, Beno\^{i}t  and  Besacier, Laurent  and  Schwab, Didier},
  title     = {FlauBERT: Unsupervised Language Model Pre-training for French},
  booktitle = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month     = {May},
  year      = {2020},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {2479--2490},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.302}
}
```

[TALN paper](https://hal.archives-ouvertes.fr/hal-02784776/)
```
@inproceedings{le2020flaubert,
  title         = {FlauBERT: des mod{\`e}les de langue contextualis{\'e}s pr{\'e}-entra{\^\i}n{\'e}s pour le fran{\c{c}}ais},
  author        = {Le, Hang and Vial, Lo{\"\i}c and Frej, Jibril and Segonne, Vincent and Coavoux, Maximin and Lecouteux, Benjamin and Allauzen, Alexandre and Crabb{\'e}, Beno{\^\i}t and Besacier, Laurent and Schwab, Didier},
  booktitle     = {Actes de la 6e conf{\'e}rence conjointe Journ{\'e}es d'{\'E}tudes sur la Parole (JEP, 31e {\'e}dition), Traitement Automatique des Langues Naturelles (TALN, 27e {\'e}dition), Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (R{\'E}CITAL, 22e {\'e}dition). Volume 2: Traitement Automatique des Langues Naturelles},
  pages         = {268--278},
  year          = {2020},
  organization  = {ATALA}
}
```

<!-- ```
@misc{le2019flaubert,
    title={FlauBERT: Unsupervised Language Model Pre-training for French},
    author={Hang Le and Loïc Vial and Jibril Frej and Vincent Segonne and Maximin Coavoux and Benjamin Lecouteux and Alexandre Allauzen and Benoît Crabbé and Laurent Besacier and Didier Schwab},
    year={2019},
    eprint={1912.05372},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
``` -->

### licence of the models
The models can be accessed on Hugging Face and their license is listed as MIT.
