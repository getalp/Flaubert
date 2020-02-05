#!/usr/bin/env bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

set -e

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

# Extract reviews and labels and split train set into train and validation sets
python flue/extract_split_cls.py --indir $DATA_DIR/raw/cls-acl10-unprocessed --outdir $DATA_DIR/processed --do_lower $do_lower

category="books dvd music"

TOKENIZER=./tools/tokenize.sh
FASTBPE=./tools/fastBPE/fast
chmod +x $TOKENIZER
chmod +x $FASTBPE

CODES_PATH=$MODEL_DIR/codes
VOCAB_PATH=$MODEL_DIR/vocab

for cat in $category; do
    for split in train valid test; do
        if [ ! -f $DATA_DIR/processed/$cat/${split}.tsv ]; then
            awk -F '\t' '{ print $1}' $DATA_DIR/processed/$cat/${split}_0.tsv \
            | $TOKENIZER 'fr' \
            > $DATA_DIR/processed/$cat/${split}.x

            awk -F '\t' '{ print $2}' $DATA_DIR/processed/$cat/${split}_0.tsv \
            > $DATA_DIR/processed/$cat/${split}.label

            $FASTBPE applybpe $DATA_DIR/processed/$cat/${split}.s1 $DATA_DIR/processed/$cat/${split}.x $CODES_PATH
            python preprocess.py $VOCAB_PATH $DATA_DIR/processed/$cat/$split.s1

            paste $DATA_DIR/processed/$cat/${split}.x $DATA_DIR/processed/$cat/${split}.label > $DATA_DIR/processed/$cat/${split}.tsv

            rm $DATA_DIR/processed/$cat/${split}_0.tsv $DATA_DIR/processed/$cat/${split}.x

            echo "Finished processing ${split} and saved to $DATA_DIR/processed/$cat."
        else
        echo 'Data has already been processed.'
        fi
    done
    echo 'Finished preparing data.'
done