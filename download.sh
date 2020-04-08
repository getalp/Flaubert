#!/bin/bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

# Script to download data from the following corpora:
# # (Corpus - corpus name to run the script)
# # 1. Wikipedia - wiki 
# # 2. Project Gutenberg - gutenberg
# # 3. EuroParl - europarl
# # 4. News Crawl - news_crawl
# # 5. News Commentary - news_commentary
# # 6. Common Crawl - common_crawl
# # 7. Other corpus as provided in the link

# Syntax to run this script:
# # ./download.sh <corpus_name> <language>
# # ./download.sh <corpus_name> <language> <corpus_date> (only for Wikipedia dumps)
# # ./download.sh <link_to_download>
# The dowloaded data is saved under the directory DATA_DIR (first argument)

set -e

# Parameters needed to specify when running script
DATA_DIR=$1
corpus=$2 # corpus_name (wiki/gutenberg/.../link to download data)
lg=$3 # input language
latest=$4 # '' or dump date in case of downloading Wikipedia 
link=$5 # download link in case corpus is not supported

# Check number of arguments
if [ $# -ge 3 ]
then
    echo "Running script ..."
else
    echo "3 arguments must be provided!"
    exit 1
fi

# Tools paths
TOOLS_PATH='tools'

# Path to save data
DATA_RAW=$DATA_DIR/raw/"$lg"_"$corpus"
# Create directory
mkdir -p $DATA_RAW


# DOWNLOAD CORPORA
if [ "$corpus" == "wiki" ]; then
    # Wiki dump name and link
    WIKI_DUMP_NAME=${lg}wiki-$latest-pages-articles-multistream.xml.bz2
    WIKI_DUMP_LINK=https://dumps.wikimedia.org/${lg}wiki/$latest/$WIKI_DUMP_NAME

    # download Wikipedia dump
    echo "***** Downloading $lg $corpus dump from $WIKI_DUMP_LINK *****"
    wget -c $WIKI_DUMP_LINK -P $DATA_RAW
    echo "Downloaded $WIKI_DUMP_NAME and saved to $DATA_RAW/$WIKI_DUMP_NAME."

elif [ "$corpus" == "gutenberg" ]; then
    # Download Gutenberg corpus
    echo "***** Downloading $lg $corpus *****"
    PYTHONIOENCODING=UTF-8 python3 $TOOLS_PATH/gutenberg_downloader.py -lang $lg -indir $DATA_RAW -update_url 1
    echo "Downloaded $lg $corpus to $DATA_RAW."

elif [ "$corpus" == "europarl" ]; then
    # Supported languages
    supported_lgs="cs de en es fi lt pl pt"

    if [ "$lg" == "fr" ]; then
        download_link=http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/europarl-v7.fr.gz
    elif [[ $supported_lgs =~ (^|[[:space:]])"$lg"($|[[:space:]]) ]]; then
        download_link=http://www.statmt.org/europarl/v9/training-monolingual/europarl-v9.$lg.gz
    else
        echo "Language not supported for the corpus $corpus." 
        rmdir $DATA_RAW
    fi

    # download data
    wget -c $download_link -P $DATA_RAW

elif [ "$corpus" == "news_commentary" ]; then
    # Supported languages
    supported_lgs="ar cs de en es fr hi id it ja kk nl pt ru zh"

    if [[ $supported_lgs =~ (^|[[:space:]])"$lg"($|[[:space:]]) ]]; then
        download_link=http://data.statmt.org/news-commentary/v14/training-monolingual/news-commentary-v14.$lg.gz
        # download data
        wget -c $download_link -P $DATA_RAW
    else
        echo "Language not supported for the corpus $corpus."
        rmdir $DATA_RAW 
    fi

elif [ "$corpus" == "news_crawl" ]; then
    supported_lgs="cs de en fi gu kk lt ru zh fr"
    if [[ $supported_lgs =~ (^|[[:space:]])"$lg"($|[[:space:]]) ]]; then
        # download data
        wget -c -r -l1 -nd -np -P $DATA_RAW -H -t1 -N -A.gz -erobots=off http://data.statmt.org/news-crawl/$lg/ 
    else
        echo "Language not supported for the corpus $corpus." 
        rmdir $DATA_RAW
    fi

elif [ "$corpus" == "common_crawl" ]; then
    supported_lgs="ar en es fr ru zh"
    if [[ $supported_lgs =~ (^|[[:space:]])"$lg"($|[[:space:]]) ]]; then
        # download data
        wget -c http://data.statmt.org/ngrams/deduped2017/$lg.deduped.xz -P $DATA_RAW

        # extract data
        echo 'Start extracting ...'
        unxz -k -T 0 $DATA_RAW/$lg.deduped.xz
        echo "Saved raw and extracted data to $DATA_RAW"
    else
        echo "Language not supported for the corpus $corpus." 
        rmdir $DATA_RAW
    fi

else
    if [ "$link" == "cirrus" ]; then
        echo "Download data from cirrus dumps"
        wget -c https://dumps.wikimedia.org/other/cirrussearch/current/"$lg$corpus"-$latest-cirrussearch-content.json.gz -P $DATA_RAW
        echo "Saved data to $DATA_RAW."
    else
        echo "Download data from supported links in $corpus" 
        wget -c $link -P $DATA_RAW
        echo "Downloaded data and saved to $DATA_RAW."
    fi
fi
