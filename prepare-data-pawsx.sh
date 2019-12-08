#! bin/bash
# Hang Le (hangtp.le@gmail.com)

set -e

DATA_DIR=$1

python extract_pawsx.py --indir $DATA_DIR/raw/x-final --outdir $DATA_DIR/processed

# data paths
DATA_PATH=$DATA_DIR/processed
TOKENIZER=tools/tokenize.sh
chmod +x $TOKENIZER

## Clean text for huggingface head
for split in train dev test; do
    awk -F '\t' '{ print $1}' $DATA_PATH/${split}_0.tsv \
    > $DATA_PATH/${split}.label

    awk -F '\t' '{ print $2}' $DATA_PATH/${split}_0.tsv \
    > $DATA_PATH/${split}.n1

    awk -F '\t' '{ print $3}' $DATA_PATH/${split}_0.tsv \
    > $DATA_PATH/${split}.n2

    awk -F '\t' '{ print $4}' $DATA_PATH/${split}_0.tsv \
    | awk '{gsub(/\"/,"")};1' \
    | sed -e 's/\(.*\)/\L\1/' \
    | $TOKENIZER fr \
    > $DATA_PATH/${split}.x1

    awk -F '\t' '{ print $5}' $DATA_PATH/${split}_0.tsv \
    | awk '{gsub(/\"/,"")};1' \
    | sed -e 's/\(.*\)/\L\1/' \
    | $TOKENIZER fr \
    > $DATA_PATH/${split}.x2

    paste $DATA_PATH/${split}.label $DATA_PATH/${split}.n1 $DATA_PATH/${split}.n2 $DATA_PATH/${split}.x1 $DATA_PATH/${split}.x2 > $DATA_PATH/${split}.tsv
    rm $DATA_PATH/${split}_0.tsv $DATA_PATH/${split}.label $DATA_PATH/${split}.n1 $DATA_PATH/${split}.n2 $DATA_PATH/${split}.x1 $DATA_PATH/${split}.x2

    echo "Finished processing ${split} and saved to $DATA_PATH."
done
echo 'Finished preparing data.'