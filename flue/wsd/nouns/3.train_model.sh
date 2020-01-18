#!/bin/bash

set -e

SCRIPT_PATH="$(dirname "$0")"/disambiguate/train.sh
DATA_PATH="$(dirname "$0")"/prepared_data

#FLAUBERT_PATH=

TARGET_DIRECTORY="$(dirname "$0")"/model


rm -rf $TARGET_DIRECTORY

$SCRIPT_PATH --data_path $DATA_PATH --model_path $TARGET_DIRECTORY --batch_size 25 --token_per_batch 2000 --update_frequency 4 --ensemble_count 1 --epoch_count 20 --eval_frequency 9999999 --input_auto_model flaubert --input_auto_path $FLAUBERT_PATH --encoder_type transformer --encoder_transformer_hidden_size 3072 --encoder_transformer_layers 6 --encoder_transformer_heads 12 --encoder_transformer_dropout 0.1 --encoder_transformer_positional_encoding false --encoder_transformer_scale_embeddings false

