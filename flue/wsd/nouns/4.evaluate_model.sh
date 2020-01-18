#!/bin/bash

set -e

SCRIPT_PATH="$(dirname "$0")"/disambiguate/evaluate.sh
DATA_PATH="$(dirname "$0")"/prepared_data
WEIGHTS_PATH="$(dirname "$0")"/model/model_weights_wsd0
CORPUS_PATH="$(dirname "$0")"/corpus/semeval2013task12.fr.xml

$SCRIPT_PATH --data_path $DATA_PATH --weights $WEIGHTS_PATH --corpus $CORPUS_PATH --lowercase true --sense_compression_hypernyms false --filter_lemma false --clear_text false

