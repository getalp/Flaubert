#!/usr/bin/env bash
# Hang Le (hangtp.le@gmail.com)
# Modified from
# https://github.com/facebookresearch/XLM/blob/master/prepare-xnli.sh
# Original copyright is appended below.
# 
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# This script is meant to prepare data to reproduce XNLI experiments
# Just modify the "code" and "vocab" path for your own model
#
# Input parameters
DATA_DIR=$1
MODEL_DIR=$2
do_lower=$3

# Check number of arguments
if [ $# -eq 3 ]
then
    echo "Running script ..."
else
    echo "3 arguments must be provided!"
    exit 1
fi

DATA_PATH=$DATA_DIR/processed

python flue/extract_xnli.py --indir $DATA_PATH --do_lower $do_lower

set -e
lg=fr

# data paths
TOKENIZER=./tools/tokenize.sh
FASTBPE=./tools/fastBPE/fast
chmod +x $TOKENIZER
chmod +x $FASTBPE

CODES_PATH=$MODEL_DIR/codes
VOCAB_PATH=$MODEL_DIR/vocab

## Clean text
for split in train valid test; do
    awk -F '\t' '{ print $1}' $DATA_PATH/${split}_0.xlm.tsv \
    | awk '{gsub(/\"/,"")};1' \
    | $TOKENIZER $lg \
    > $DATA_PATH/${split}.x1

    awk -F '\t' '{ print $2}' $DATA_PATH/${split}_0.xlm.tsv \
    | awk '{gsub(/\"/,"")};1' \
    | $TOKENIZER $lg \
    > $DATA_PATH/${split}.x2

    awk -F '\t' '{ print $3}' $DATA_PATH/${split}_0.xlm.tsv \
    > $DATA_PATH/${split}.label

    echo "Finished processing ${split} and saved to $DATA_PATH."
done
echo 'Finished preparing data.'

# apply BPE codes and binarize the GLUE corpora
for splt in train valid test; do
    echo "BPE-rizing $splt..."
    $FASTBPE applybpe $DATA_PATH/$splt.s1 $DATA_PATH/$splt.x1 $CODES_PATH
    python preprocess.py $VOCAB_PATH $DATA_PATH/$splt.s1
    rm $DATA_PATH/$splt.x1

    $FASTBPE applybpe $DATA_PATH/$splt.s2 $DATA_PATH/$splt.x2 $CODES_PATH
    python preprocess.py $VOCAB_PATH $DATA_PATH/$splt.s2
    rm $DATA_PATH/$splt.x2
done