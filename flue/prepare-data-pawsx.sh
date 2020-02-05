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

python flue/extract_pawsx.py --indir $DATA_DIR/raw/x-final --outdir $DATA_DIR/processed --do_lower $do_lower

# data paths
TOKENIZER=./tools/tokenize.sh
FASTBPE=./tools/fastBPE/fast
chmod +x $TOKENIZER
chmod +x $FASTBPE

CODES_PATH=$MODEL_DIR/codes
VOCAB_PATH=$MODEL_DIR/vocab

## Clean text
for split in train valid test; do
    awk -F '\t' '{ print $1}' $DATA_DIR/processed/${split}_0.tsv \
    > $DATA_DIR/processed/${split}.label

    awk -F '\t' '{ print $2}' $DATA_DIR/processed/${split}_0.tsv \
    > $DATA_DIR/processed/${split}.n1

    awk -F '\t' '{ print $3}' $DATA_DIR/processed/${split}_0.tsv \
    > $DATA_DIR/processed/${split}.n2

    awk -F '\t' '{ print $4}' $DATA_DIR/processed/${split}_0.tsv \
    | awk '{gsub(/\"/,"")};1' \
    | $TOKENIZER fr \
    > $DATA_DIR/processed/${split}.x1

    awk -F '\t' '{ print $5}' $DATA_DIR/processed/${split}_0.tsv \
    | awk '{gsub(/\"/,"")};1' \
    | $TOKENIZER fr \
    > $DATA_DIR/processed/${split}.x2

    $FASTBPE applybpe $DATA_DIR/processed/${split}.s1 $DATA_DIR/processed/${split}.x1 $CODES_PATH
    python preprocess.py $VOCAB_PATH $DATA_DIR/processed/$split.s1

    $FASTBPE applybpe $DATA_DIR/processed/${split}.s2 $DATA_DIR/processed/${split}.x2 $CODES_PATH
    python preprocess.py $VOCAB_PATH $DATA_DIR/processed/$split.s2

    paste $DATA_DIR/processed/${split}.label $DATA_DIR/processed/${split}.n1 $DATA_DIR/processed/${split}.n2 $DATA_DIR/processed/${split}.x1 $DATA_DIR/processed/${split}.x2 > $DATA_DIR/processed/${split}.tsv
    rm $DATA_DIR/processed/${split}_0.tsv $DATA_DIR/processed/${split}.n1 $DATA_DIR/processed/${split}.n2 $DATA_DIR/processed/${split}.x1 $DATA_DIR/processed/${split}.x2

    echo "Finished processing ${split} and saved to $DATA_DIR/processed."
done
echo 'Finished preparing data.'