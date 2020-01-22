#!/bin/bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

# Script to extract and preprocess text, 
#     including sanitization, Moses tokenization,
#     and sentence segmentation using NLTK.

# Syntax to run this script:
# ./preprocess <corpus_name> <language>

set -e

# Parameters needed to specify to run this script
DATA_DIR=$1
corpus=$2 # corpus_name (wiki / gutenberg / europarl etc.)
lg=$3 # input language

# Check number of arguments
if [ $# -eq 3 ]
then
    echo "Running script ..."
else
    echo "3 arguments must be provided!"
    exit 1
fi

# Path to raw and processed data
DATA_PATH_RAW=$DATA_DIR/raw/"$lg"_"$corpus"
DATA_PATH_CLEAN=$DATA_DIR/processed/"$lg"_"$corpus"
output=$DATA_PATH_CLEAN/$lg.$corpus.processed.nc

# Tools paths
TOOLS_PATH='tools'

# Tokenizer, cleaner and sentence splitter
TOKENIZER=$TOOLS_PATH/tokenize.sh
CLEANER=$TOOLS_PATH/clean_text.py
SENT_SPLITTER=$TOOLS_PATH/split_sentences.py

# Create folder to save processed data
# exit if raw data is not available in folder yet
if [ -d "$DATA_PATH_RAW" ]; then
    mkdir -p $DATA_PATH_CLEAN
else
    echo "Raw data of $lg $corpus is not available in $DATA_PATH_RAW yet."
    exit 1
fi


function preprocess {
    local cp=$1
    local path_raw=$2
    local path_clean=$3

    if [ "$cp" == "common_crawl" ]; then
        f=$DATA_PATH_RAW/$lg.deduped
        fo=$DATA_PATH_CLEAN/$lg.deduped.processed.nc

        echo "Processing $f ..."
        
        # Apply aggressive heuristics to filter data from Common Crawl
        if [ ! -f  "$fo" ]; then
            # Clean, tokenize and split sentences
            python $CLEANER -p "$f" \
            | grep -P -v '^\s*$' \
            | grep -P '.{50,}' \
            | grep -P -v '[|\t\[\]\{\}]' \
            | grep -P -v '\\{2,}' \
            | grep -P -v '\(en savoir plus\)' \
            | grep -P -v '(?:([-[\](){}><]+ *\w* *[-[\](){}><]+) *\w* *){5,}' \
            | grep -P -o '(^\p{Lu}|(?<=[.!?]\s))\p{Lu}.{50,}(\w\.|\s\!|\s\?)+' | grep -P -v '\d+ Fax|Tel \(' \
            | grep -P -v '[eE]mail|[fF]ax|[tT]éléphone|[tT]el|[cC]ontact|[i|I]nfo *[@:]+' \
            | grep -P -v '^:' \
            | grep -P -v '(\/ ){3,}|(\/){3,}' \
            | grep -P -v '[A-Za-z0-9]{25,}' \
            | grep -P -v '(\w{4,}\d{2,})|(\d{2,}\w{4,})' \
            | grep -P '^(?!.*(.)\1{5,})' \
            | grep -P '^(?!.*(..)\1{5,})' \
            | perl -CSD -Mutf8 -pe 's/\p{Sk}+|\p{So}+|\p{Cn}+|\p{Co}+|\p{Cs}+|\p{M}+|\p{Lo}+//g' \
            | $TOKENIZER $lg \
            | python $SENT_SPLITTER \
            | grep -P -v '(\w ){10,}' \
            | grep -P -v '(\w |\w\w ){10,}' \
            | grep -P -v '^(\/ [>.*\d])' \
            | grep -P -v '^(: \d+)|^(: [()"-:+])' \
            | grep -P -v '^\s*$' \
            | grep -P '.{50,}' \
            > "$fo"
            echo "Finished cleaning and tokenizing data. Processed files are saved in $path_clean."
        else
            echo "Data has already been processed and saved in $path_clean."
        fi

    elif [[ "$cp" == "wiki"* ]]; then
        echo "Corpus: $cp."
        for f in $path_raw/*; do
            # Extract file names
            fo=$path_clean/$(basename ${f%.*})

            # Check if processed files are available
            if [ ! -f  "$fo.processed.nc" ]; then
                # Clean, tokenize and split sentences
                python $CLEANER -p "$f" \
                | grep -P -v '^\s*$' \
                | grep -P '.{50,}' \
                | grep -P -v '[|\t\[\]\{\}]' \
                | grep -P -v '\(en savoir plus\)' \
                | grep -P -v '\d+ Fax|Tel \(' \
                | grep -P -v '[eE]mail|[fF]ax|[tT]éléphone|[tT]el|[cC]ontact|[i|I]nfo *[@:]+' \
                | grep -P '^(?!.*(.)\1{5,})' \
                | grep -P '^(?!.*(..)\1{5,})' \
                | grep -P -v '[A-Za-z0-9]{30,}' \
                | grep -P -v '(\w{4,}\d{3,})|(\d{3,}\w{4,})' \
                | grep -v "<br|br/>" \
                | grep -P -v 'noinclude|pagequality|user=|\{\{|\}\}|\\|<\/\w+>|\|\w*\|\/>|<\w+>|<section|style=' \
                | perl -CSD -Mutf8 -pe 's/\p{Sk}+|\p{So}+|\p{Cn}+|\p{Co}+|\p{Cs}+|\p{M}+//g' \
                | $TOKENIZER $lg \
                | python $SENT_SPLITTER \
                | grep -P -v '(\w ){15,}' \
                | grep -P -v '(\w |\w\w ){10,}' \
                | grep -P -v '^\s*$' \
                | grep -P '.{20,}' \
                > "$fo.processed.nc"
                echo "Finished cleaning and tokenizing data. Processed files are saved in $path_clean."
            else
                echo "Data has already been processed and saved in $path_clean."
            fi
        done

    else
        # for other corpora
        for f in $path_raw/*; do
            # Extract file names
            fo=$path_clean/$(basename ${f%.*})

            # Check if processed files are available
            if [ ! -f  "$fo.processed.nc" ]; then
                # Clean, tokenize and split sentences
                python $CLEANER -p "$f" \
                | grep -P -v '^\s*$' \
                | grep -P '.{50,}' \
                | grep -P -v '[|\t\[\]\{\}]' \
                | grep -P -v '\d+ Fax|Tel \(' \
                | grep -P -v '[eE]mail|[fF]ax|[tT]éléphone|[tT]el|[cC]ontact|[i|I]nfo *[@:]+' \
                | grep -P -v '\(en savoir plus\)' \
                | grep -P '^(?!.*(.)\1{5,})' \
                | grep -P '^(?!.*(..)\1{5,})' \
                | grep -P -v '[A-Za-z0-9]{30,}' \
                | grep -P -v '(\w{4,}\d{3,})|(\d{3,}\w{4,})' \
                | perl -CSD -Mutf8 -pe 's/\p{Sk}+|\p{So}+|\p{Cn}+|\p{Co}+|\p{Cs}+|\p{M}+|\p{Lo}+//g' \
                | $TOKENIZER $lg \
                | python $SENT_SPLITTER \
                | grep -P -v '(\w ){15,}' \
                | grep -P -v '(\w |\w\w ){10,}' \
                | grep -P -v '^:' \
                | grep -P -v '^\s*$' \
                | grep -P '.{20,}' \
                > "$fo.processed.nc"

                echo "Finished cleaning and tokenizing data. Processed files are saved in $path_clean."
            else
                echo "Data has already been processed and saved in $path_clean."
            fi
        done

    fi
}


# PREPROCESS CORPORA
if [ "$corpus" == "wiki" ]; then
    
    if [ ! -f $output ]; then
        echo "***** Cleaning, tokenizing and segmenting $lg Wikipedia dump ... *****"
        python $TOOLS_PATH/wikiextractor/WikiExtractor.py $DATA_PATH_RAW/*.bz2 --processes 8 -q -o - \
        | grep -P -v '^\s*$' \
        | grep -v "^<doc id=" \
        | grep -v "</doc>\$" \
        | grep -v "<br>" \
        | grep -P '^(?!.*(.)\1{5,})' \
        | grep -P '^(?!.*(..)\1{5,})' \
        | perl -CSD -Mutf8 -pe 's/\p{Sk}+|\p{So}+|\p{Cn}+|\p{Co}+|\p{Cs}+|\p{M}+//g' \
        | python $CLEANER -i 1 \
        | $TOKENIZER $lg \
        | python $SENT_SPLITTER \
        | grep -P -v '^\s*$' \
        | grep -P -v '(\w ){15,}' \
        | grep -P -v '(\w |\w\w ){10,}' \
        | grep -P '.{10,}' \
        > $output
        echo "***** Saved preprocessed data to $output *****"
    else
        echo "Data has been preprocessed and saved in $output."
    fi
    

elif [ "$corpus" == "gutenberg" ]; then

    if [ ! -f $output ]; then 
        echo "***** Cleaning and tokenizing $lg Project Gutenberg *****"
        python $TOOLS_PATH/gutenberg_cleaner.py -indir $DATA_PATH_RAW -outdir $DATA_PATH_CLEAN

        cat $DATA_PATH_CLEAN/*.txt > $DATA_PATH_CLEAN/pre.all

        grep -P -v '^\s*$' $DATA_PATH_CLEAN/pre.all \
        | grep -P '.{20,}' \
        | grep -P -v '[|\t\[\]\{\}]' \
        | grep -P -v '\d+ Fax|Tel \(' \
        | grep -P '^(?!.*(.)\1{5,})' \
        | grep -P '^(?!.*(..)\1{5,})' \
        | grep -P -v '[A-Za-z0-9]{30,}' \
        | grep -P -v '(\w{4,}\d{3,})|(\d{3,}\w{4,})' \
        | perl -CSD -Mutf8 -pe 's/\p{Sk}+|\p{So}+|\p{Cn}+|\p{Co}+|\p{Cs}+|\p{M}+//g' \
        | $TOKENIZER $lg \
        | grep -P -v '(\w ){15,}' \
        | grep -P -v '(\w |\w\w ){10,}' \
        | grep -P -v '^\s*$' \
        | grep -P '.{20,}' \
        > $output

        rm $DATA_PATH_CLEAN/*.txt
        echo "Removed txt files."
        echo "***** Saved preprocessed data to $output *****"

    else
        echo "Data has been preprocessed and saved in $output."
    fi

else
    preprocess $corpus $DATA_PATH_RAW $DATA_PATH_CLEAN
fi