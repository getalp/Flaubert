#!/usr/bin/env bash
# Hang Le (hangtp.le@gmail.com)
# Modified from
# https://github.com/facebookresearch/XLM/blob/master/get-data-xnli.sh
# Original copyright is appended below.
# 
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-xnli.sh
#
set -e

lg=fr

DATA_DIR=$1
# data paths
DATA_RAW=$DATA_DIR/raw
DATA_PROC=$DATA_DIR/processed

f=XNLI-1.0
ft=XNLI-MT-1.0

URLPATH=https://dl.fbaipublicfiles.com/XNLI/${f}.zip
URLPATH_TRAIN=https://dl.fbaipublicfiles.com/XNLI/${ft}.zip

mkdir -p $DATA_RAW
mkdir -p $DATA_PROC

# Download data
wget -c $URLPATH -P $DATA_RAW
wget -c $URLPATH_TRAIN -P $DATA_RAW

# unzip data
if [ ! -d $DATA_RAW/$f ]; then
    echo "Unzipping data ..."
    unzip $DATA_RAW/${f}.zip -d $DATA_RAW
    unzip $DATA_RAW/${ft}.zip -d $DATA_RAW
else
    echo "Files have been already unzipped."
fi

echo "***** Preparing valid and test data *****"

# 1st row
echo -e "premise\thypo\tlabel" > $DATA_PROC/$lg.raw.valid
echo -e "premise\thypo\tlabel" > $DATA_PROC/$lg.raw.test

# label
awk -v lg=$lg '$1==lg' $DATA_RAW/$f/xnli.dev.tsv  | cut -f2 > $DATA_PROC/dev.f2
awk -v lg=$lg '$1==lg' $DATA_RAW/$f/xnli.test.tsv | cut -f2 > $DATA_PROC/test.f2

# premise/hypothesis
awk -v lg=$lg '$1==lg' $DATA_RAW/$f/xnli.dev.tsv  | cut -f7 > $DATA_PROC/dev.f7
awk -v lg=$lg '$1==lg' $DATA_RAW/$f/xnli.dev.tsv  | cut -f8 > $DATA_PROC/dev.f8

awk -v lg=$lg '$1==lg' $DATA_RAW/$f/xnli.test.tsv | cut -f7 > $DATA_PROC/test.f7
awk -v lg=$lg '$1==lg' $DATA_RAW/$f/xnli.test.tsv | cut -f8 > $DATA_PROC/test.f8

paste $DATA_PROC/dev.f7  $DATA_PROC/dev.f8  $DATA_PROC/dev.f2  >> $DATA_PROC/$lg.raw.valid
paste $DATA_PROC/test.f7 $DATA_PROC/test.f8 $DATA_PROC/test.f2 >> $DATA_PROC/$lg.raw.test

rm $DATA_PROC/*.f2 
rm $DATA_PROC/*.f7 
rm $DATA_PROC/*.f8

# Translated train set from English
echo "***** Preparing train set in $lg ******"
echo -e "premise\thypo\tlabel" > $DATA_PROC/$lg.raw.train
sed '1d'  $DATA_RAW/$ft/multinli/multinli.train.$lg.tsv | cut -f1 > $DATA_PROC/train.f1
sed '1d'  $DATA_RAW/$ft/multinli/multinli.train.$lg.tsv | cut -f2 > $DATA_PROC/train.f2
sed '1d'  $DATA_RAW/$ft/multinli/multinli.train.$lg.tsv | cut -f3 | sed 's/contradictory/contradiction/g' > $DATA_PROC/train.f3
paste $DATA_PROC/train.f1 $DATA_PROC/train.f2 $DATA_PROC/train.f3 >> $DATA_PROC/$lg.raw.train

rm $DATA_PROC/train.f1 $DATA_PROC/train.f2 $DATA_PROC/train.f3

echo "Finished."