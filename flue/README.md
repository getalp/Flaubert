# FLUE: French Language Understanding Evaluation

# News

- May 13th 2020: Code and pre-trained models for [Dependency Parsing](#42-dependency-parsing) are available!

# Introduction

**FLUE** is an evaluation setup for French NLP systems similar to the popular GLUE benchmark. The goal is to enable further reproducible experiments in the future and to share models and progress on the French language. The tasks and data are obtained from existing works, please refer to our [Flaubert paper](https://arxiv.org/abs/1912.05372) for a complete list of references.

On this page we describe the tasks and provide examples of usage.

A leaderboard will be updated frequently [here](https://github.com/getalp/Flaubert/tree/master/flue/leaderboard).
 
# Table of Contents
**1. [Text Classification](#1-text-classification-cls)**    
**2. [Paraphrasing](#2-paraphrasing-paws-x)**  
**3. [Natural Language Inference](#3-natural-language-inference-xnli)**     
**4. [Parsing](#4-parsing)**  
&nbsp;&nbsp;&nbsp;&nbsp;4.1. [Constituency Parsing](#41-constituency-parsing)   
&nbsp;&nbsp;&nbsp;&nbsp;4.2. [Dependency Parsing](#42-dependency-parsing)   
**5. [Word Sense Disambiguation](#5-word-sense-disambiguation)**    
&nbsp;&nbsp;&nbsp;&nbsp;5.1. [Verb Sense Disambiguation](#51-verb-sense-disambiguation)     
&nbsp;&nbsp;&nbsp;&nbsp;5.2. [Noun Sense Disambiguation](#52-noun-sense-disambiguation)   
<!-- **6. [Named Entity Recognition](#6.-Named-Entity-Recognition)**     
**7. [Question Answering](#7.-Question-Answering)** -->

In the following, you should replace `$DATA_DIR` with a location on your computer, e.g. `~/data/cls`, `~/data/pawsx`, `~/data/xnli`, etc. depending on the task. Raw data is downloaded and saved to `$DATA_DIR/raw` by running the below command
```bash
bash get-data-${task}.sh $DATA_DIR
```
where `${task}` is either `cls, pawsx, xnli`.

`$MODEL_DIR` denotes the path to where you save the pretrainded FlauBERT model, which contains 3 files:
- `*.pth`: FlauBERT's pretrained model.
- `codes`: BPE codes learned on the training data.
- `vocab`: BPE vocabulary file.

You can download these pretrained models from [here](https://zenodo.org/record/3626826).

# 1. Text Classification (CLS)

## Task description
This is a binary classification task. It consists in classifying Amazon reviews for three product categories: *books*, *DVD*, and *music*. Each sample contains a review text and the associated rating from 1 to 5 stars. Reviews rated above 3 is labeled as *positive*, and those rated less than 3 is labeled as *negative*.

## Dataset
The train and test sets are balanced, including around 1k positive and 1k negative reviews for a total of 2k reviews in each dataset. We take the French portion to create the binary text classification task in FLUE and report the accuracy on the test set.

**Download**:
```bash
bash flue/get-data-cls.sh $DATA_DIR
```
The ouput files (train and test sets) obtained from the above script are: `$DATA_DIR/raw/cls-acl10-unprocessed/${lang}/${category}/${split}.review`, where:
- `${lang}` includes `de, en, fr, jp`
- `${category}` includes `books, dvd, music`
- `${split}` includes `train, test`

In this task, we use the related datasets for French (`fr`).

## Example
### a. Finetuning FlauBERT with Facebook's XLM library
In this example, we describe how to finetune FlauBERT using the [XLM](https://github.com/facebookresearch/XLM) library.

We split the `train` set into `train` and `valid` sets for training and validation (the default validation ratio is set to be `0.2`. You can change this ratio in the `flue/extract_split_cls.py` script).

#### Preprocess data
To finetune FlauBERT on this task using the XLM library, we need to do some data pre-processing steps as follows: 
- (1) Clean and tokenize text using [Moses](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer) tokenizer
- (2) Apply BPE codes and vocabulary learned in pre-training (we used [fastBPE](https://github.com/glample/fastBPE))
- (3) Binarize data

Run the following command to split the training set and perform the above preprocessing steps:
```bash
bash flue/prepare-data-cls.sh $DATA_DIR $MODEL_DIR $do_lower
```
where `$do_lower` should be set to `True` (or `true, on, 1`) if you use the uncased (lower-case) pretrained model, otherwise it should be set to `False` (or `false, off, 0`).

#### Finetune
Run the below command to finetune Flaubert with the parameters from the configuration file `flue/examples/cls_books_lr5e6_xlm_base_cased.cfg` as an example.

```bash
config='flue/examples/cls_books_lr5e6_xlm_base_cased.cfg'
source $config

python flue/flue_xnli.py --exp_name $exp_name \
                        --exp_id $exp_id \
                        --dump_path $dump_path  \
                        --model_path $model_path  \
                        --data_path $data_path  \
                        --dropout $dropout \
                        --transfer_tasks $transfer_tasks \
                        --optimizer_e adam,lr=$lre \
                        --optimizer_p adam,lr=$lrp \
                        --finetune_layers $finetune_layer \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
```

### b. Finetuning FlauBERT with Hugging Face's Transformers library
<!-- You should follow the instructions [here](https://github.com/getalp/Flaubert#11-using-flaubert-with-hugging-faces-transformers) to clone our Transformer repo forked from [Hugging Face's Transformer](https://github.com/huggingface/transformers), in which we added our Flaubert's classes. We will update the instructions when our model is integrated into official Hugging Face's Transformer library. -->

<!-- ```bash
pip install git+https://github.com/formiel/transformers.git --upgrade
``` -->

#### Preprocess data
Run the below command to prepare data for finetuning. The tokenization (Moses and BPE) is handled later using `FlaubertTokenizer` class in the fine-tuning script.

```bash
python flue/extract_split_cls.py --indir $DATA_DIR/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/processed \
                                 --do_lower true \
                                 --use_hugging_face true
```

#### Finetune
<!-- Run the below command to finetune Flaubert using the Transformers repo from [above](#b.-Finetuning-FlauBERT-with-Hugging-Face's-transformers-library), where `~/transformers` should be replaced by the local path where you save the forked repo. -->
Run the below command to finetune Flaubert using [Hugging Face's Transformers](https://github.com/huggingface/transformers) library.

```bash
config='flue/examples/cls_books_lr5e6_hf_base_uncased.cfg'
source $config

python ~/transformers/examples/run_glue.py \
                                        --data_dir $data_dir \
                                        --model_type flaubert \
                                        --model_name_or_path $model_name_or_path \
                                        --task_name $task_name \
                                        --output_dir $output_dir \
                                        --max_seq_length 512 \
                                        --do_train \
                                        --do_eval \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --save_steps $save_steps \
                                        --fp16 \
                                        --fp16_opt_level O1 \
                                        |& tee output.log
```
You can add `CUDA_VISIBLE_DEVICES=0,1` in the beginning of the above running command to run the fine-tuning on 2 GPUs for example.

# 2. Paraphrasing (PAWS-X)
## Task description
The task consists in identifying whether the two sentences in a pair are semantically equivalent or not.

## Dataset
The train set includes 49.4k examples, the dev and test sets each comprises nearly 2k examples. We take the related datasets for French to perform the paraphrasing task and report the accuracy on the test set.

**Download**:
```bash
bash flue/get-data-pawsx.sh $DATA_DIR
```
The ouput files obtained from the above script are: `$DATA_DIR/raw/x-final/${lang}`, where `${lang}` includes `de, en, es, fr, ja, ko, zh`. Each folder comprises 3 files: `translated_train.tsv`, `dev_2k.tsv`, and `test_2k.tsv`

In this task, we use the related datasets for French (`fr`).

## Example
### a. Finetuning FlauBERT with Facebook's XLM library

#### Preprocess data
The preprocessing includes 3 steps as described in the example in the [text classification task](#1.-Text-Classification-(CLS)).

```bash
bash flue/prepare-data-pawsx.sh $DATA_DIR $MODEL_DIR $do_lower
```

#### Finetune
Run the below command to finetune Flaubert with the parameters from an input configuration file.

```bash
config='flue/examples/pawsx_lr5e6_xlm_base_cased.cfg'
source $config

python flue/flue_xnli.py --exp_name $exp_name \
                        --exp_id $exp_id \
                        --dump_path $dump_path  \
                        --model_path $model_path  \
                        --data_path $data_path  \
                        --dropout $dropout \
                        --transfer_tasks $transfer_tasks \
                        --optimizer_e adam,lr=$lre \
                        --optimizer_p adam,lr=$lrp \
                        --finetune_layers $finetune_layer \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
```

### b. Finetuning FlauBERT with Hugging Face's transformers library

#### Preprocess data
Run the below command to prepare data for finetuning. The tokenization (Moses and BPE) is handled later using `FlaubertTokenizer` class in the fine-tuning script.

```bash
python flue/extract_pawsx.py --indir ~/Data/FLUE/pawsx/raw/x-final \
                             --outdir ~/Data/FLUE/pawsx/processed \
                             --use_hugging_face True
```

#### Finetune
<!-- Run the below command to finetune Flaubert using the Transformers repo from [above](#b.-Finetuning-FlauBERT-with-Hugging-Face's-transformers-library), where `~/transformers` should be replaced by the local path where you save the forked repo. -->
Run the below command to finetune Flaubert on `PAWSX` dataset using [Hugging Face's Transformers](https://github.com/huggingface/transformers) library.

```bash
config='flue/examples/pawsx_lr5e6_hf_base_cased.cfg'
source $config

python ~/transformers/examples/run_glue.py \
                                        --data_dir $data_dir \
                                        --model_type flaubert \
                                        --model_name_or_path $model_name_or_path \
                                        --task_name $task_name \
                                        --output_dir $output_dir \
                                        --max_seq_length 512 \
                                        --do_train \
                                        --do_eval \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --save_steps $save_steps \
                                        --fp16 \
                                        --fp16_opt_level O1 \
                                        |& tee output.log
```

# 3. Natural Language Inference (XNLI)
## Task description
The Natural Language Inference (NLI) task, also known as recognizing textual entailment (RTE), is to determine whether a premise entails, contradicts or neither entails nor contradicts a hypothesis. We take the French part of the XNLI corpus to form the development and test sets for the NLI task in FLUE.

## Dataset
The train set includes 392.7k examples, the dev and test sets comprises 2.5k and 5k examples respectively. We take the related datasets for French to perform the NLI task and report the accuracy on the test set.

**Download**:
```bash
bash flue/get-data-xnli.sh $DATA_DIR
```
The output files from the above script are: `$DATA_DIR/processed/fr.raw.${split}`, where `${split}` includes `train, valid, test`.

## Example
### a. Finetuning FlauBERT with Facebook's XLM library

#### Preprocess data
The preprocessing includes 3 steps as described in the example in the [text classification task](#1.-Text-Classification-(CLS)).

```bash
bash flue/prepare-data-xnli.sh $DATA_DIR $MODEL_DIR $do_lower
```

#### Finetune
Run the below command to finetune Flaubert with the parameters from an input configuration file.

```bash
config='flue/examples/xnli_lr5e6_xlm_base_cased.cfg'
source $config

python flue/flue_xnli.py --exp_name $exp_name \
                        --exp_id $exp_id \
                        --dump_path $dump_path  \
                        --model_path $model_path  \
                        --data_path $data_path  \
                        --dropout $dropout \
                        --transfer_tasks $transfer_tasks \
                        --optimizer_e adam,lr=$lre \
                        --optimizer_p adam,lr=$lrp \
                        --finetune_layers $finetune_layer \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
```

### b. Finetuning FlauBERT with Hugging Face's transformers library
Coming soon.

# 4. Parsing

## 4.1. Constituency Parsing
The French Treebank collection is freely available for research purposes.
See [here](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php?langue=en) to download the latest version of the corpus and sign the license, and [here](http://dokufarm.phil.hhu.de/spmrl2014/) to obtain the version of the corpus used for the experiments described in the paper.

To fine-tune FlauBERT on constituency parsing on the French Treebank, see instructions [here](https://github.com/mcoavoux/self-attentive-parser).

Pretrained parsing models for both FlauBERT and CamemBERT are now available!

## 4.2. Dependency Parsing
To fine-tune FlauBERT on dependency parsing, see instructions [here](https://github.com/bencrabbe/npdependency).

Pretrained models for both FlauBERT and CamemBERT are available!

# 5. Word Sense Disambiguation
## 5.1. Verb Sense Disambiguation
To evaluate Flaubert on the French Verb Sense Disambiguation task:

  **1. Download the FrenchSemEval (FSE) dataset available [here](http://www.llf.cnrs.fr/dataset/fse/)** (called ```$FSE_DIR``` hereafter)
  
  **2. Prepare the data**
  ```python
  python prepare_data.py --data $FSE_DIR --output $DATA_DIR
  ```
  
  **3. Run the model and evaluate with ```flue_vsd.py```**
  ```python
  python flue_vsd.py --exp_name myexp --model flaubert-base-cased --data $DATA_DIR --padding 80 --batchsize 32 --device 0 --output $OUTPUT_DIR
  ```
  You can use this script to evaluate either a pretrained-model or your own model (from checkpoint). Yet It has to be one of the Flaubert/Camembert/Bert class of the Hugginface API.
  
  See further options in the ```flue/wsd/verbs/``` directory.

## 5.2. Noun Sense Disambiguation

To fine-tune Flaubert for French WSD with WordNet as sense inventory, you can follow the scripts located in the directory [wsd/nouns](wsd/nouns), which allow you to:
- Automatically download our publicly available dataset from [this address](https://zenodo.org/record/3549806)  
  → See the script [0.get_data.sh](wsd/nouns/0.get_data.sh)
- Download the `disambiguate` toolkit from [this repository](https://github.com/getalp/disambiguate)  
  → See the script [1.get_toolkit.sh](wsd/nouns/1.get_toolkit.sh)
- Prepare the training/development data from the French SemCor and French WordNet Gloss Corpus  
 → See the script [2.prepare_data.sh](wsd/nouns/2.prepare_data.sh)
- Train the neural model  (assumes that `$FLAUBERT_PATH` is the path to a Flaubert model)  
  → See the script [3.train_model.sh](wsd/nouns/3.train_model.sh)
- Evaluate the model on the French SemEval 2013 task 12 corpus  
  → See the script [4.evaluate_model.sh](wsd/nouns/4.evaluate_model.sh)
  
Once the model is trained, you can disambiguate any text using the script [5.disambiguate.sh](wsd/nouns/5.disambiguate.sh)

<!-- # 6. Named Entity Recognition
Coming soon.

# 7. Question Answering
Coming soon. -->


# Citation
If you use FlauBERT or the FLUE Benchmark for your scientific publication, or if you find the resources in this repository useful, please refer to our [paper](https://arxiv.org/abs/1912.05372):

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
