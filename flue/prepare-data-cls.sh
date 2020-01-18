#! bin/bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

set -e

# data paths
DATA_DIR=$1
do_lower=$2

# Extract reviews and labels and split train set into train and validation sets
python extract_split_cls.py --indir $DATA_DIR/raw/cls-acl10-unprocessed --outdir $DATA_DIR/processed --do_lower $do_lower

category="books dvd music"

DATA_PATH=$DATA_DIR/processed
TOKENIZER=tools/tokenize.sh
chmod +x $TOKENIZER

for cat in $category; do
    ## Clean text
    for split in train dev test; do
        if [ ! -f $DATA_PATH/$cat/${split}.tsv ]; then
            awk -F '\t' '{ print $1}' $DATA_PATH/$cat/${split}_0.tsv \
            | $TOKENIZER 'fr' \
            > $DATA_PATH/$cat/${split}.x

            awk -F '\t' '{ print $2}' $DATA_PATH/$cat/${split}_0.tsv \
            > $DATA_PATH/$cat/${split}.y

            paste $DATA_PATH/$cat/${split}.x $DATA_PATH/$cat/${split}.y > $DATA_PATH/$cat/${split}.tsv
            rm $DATA_PATH/$cat/${split}_0.tsv $DATA_PATH/$cat/${split}.x $DATA_PATH/$cat/${split}.y

            echo "Finished processing ${split} and saved to $DATA_PATH."
        else
        echo 'Data has already been processed.'
        fi
    done
    echo 'Finished preparing data.'
done
