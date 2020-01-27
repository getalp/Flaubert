#! bin/bash
# Hang Le (hangtp.le@gmail.com)
# Modified from
# https://github.com/facebookresearch/XLM
# Original copyright is appended below.
# 
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

lg=fr

# Specify parameters to run the script
DATA_DIR=$1 # path to directory where you save train, valid, and test sets

# Num tokens in 1000 units: 50 for 50000 codes, 0.1 for 100 codes
num_tokens_k=$2

file_bpe=$num_tokens_k'k'
num_tokens=$(($num_tokens_k * 1000))

# path to the fastBPE tool  
FASTBPE=tools/fastBPE/fast 

# path where processed files will be stored
OUT_PATH=$DATA_DIR/BPE/$file_bpe 

# create output path
mkdir -p $OUT_PATH

# Learn bpe codes on the training set
$FASTBPE learnbpe $num_tokens $DATA_DIR/$lg.train > $OUT_PATH/codes

# Apply BPE on train, valid, and test sets
$FASTBPE applybpe $OUT_PATH/train.$lg $DATA_DIR/$lg.train $OUT_PATH/codes
$FASTBPE applybpe $OUT_PATH/valid.$lg $DATA_DIR/$lg.valid $OUT_PATH/codes
$FASTBPE applybpe $OUT_PATH/test.$lg $DATA_DIR/$lg.test $OUT_PATH/codes

# Get the post-BPE vocabulary
cat $OUT_PATH/train.$lg | $FASTBPE getvocab - > $OUT_PATH/vocab

# Binarize data
python preprocess.py $OUT_PATH/vocab $OUT_PATH/train.$lg
python preprocess.py $OUT_PATH/vocab $OUT_PATH/valid.$lg
python preprocess.py $OUT_PATH/vocab $OUT_PATH/test.$lg