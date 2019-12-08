#! bin/bash
# Hang Le (hangtp.le@gmail.com)

set -e

# data paths
DATA_DIR=$1

python extract_split_cls.py --indir $DATA_DIR/raw/cls-acl10-unprocessed -outdir $DATA_DIR/processed

category="books dvd music"


DATA_PATH=$DATA_DIR/processed
TOKENIZER=tools/tokenize.sh
chmod +x $TOKENIZER

for cat in $category; do
    ## Clean text
    if [ ! -f $DATA_PATH/$cat/train.tsv ]; then
        for split in train dev test; do
            awk -F '\t' '{ print $1}' $DATA_PATH/$cat/${split}_0.tsv \
            | sed -e 's/\(.*\)/\L\1/' \
            | $TOKENIZER 'fr' \
            > $DATA_PATH/$cat/${split}.x

            awk -F '\t' '{ print $2}' $DATA_PATH/$cat/${split}_0.tsv \
            > $DATA_PATH/$cat/${split}.y

            paste $DATA_PATH/$cat/${split}.x $DATA_PATH/$cat/${split}.y > $DATA_PATH/$cat/${split}.tsv
            rm $DATA_PATH/$cat/${split}_0.tsv $DATA_PATH/$cat/${split}.x $DATA_PATH/$cat/${split}.y

            echo "Finished processing ${split} and saved to $DATA_PATH."
        done
        echo 'Finished preparing data.'
    else
        echo 'Data has already been processed.'
    fi
done