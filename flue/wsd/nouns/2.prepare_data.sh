#!/bin/bash

set -e

SCRIPT_PATH="$(dirname "$0")"/disambiguate/prepare_data.sh

SEMCOR_PATH="$(dirname "$0")"/corpus/semcor.fr.xml
WNGT_PATH="$(dirname "$0")"/corpus/wngt.fr.xml

TARGET_DIRECTORY="$(dirname "$0")"/prepared_data

rm -rf $TARGET_DIRECTORY

$SCRIPT_PATH --data_path $TARGET_DIRECTORY --train $SEMCOR_PATH $WNGT_PATH --dev_from_train 4000 --input_clear_text true --sense_compression_hypernyms false --lowercase true

