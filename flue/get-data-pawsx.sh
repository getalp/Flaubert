#!/usr/bin/env bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

set -e

# data paths
DATA_DIR=$1
DATA_RAW=$DATA_DIR/raw
DATA_PROC=$DATA_DIR/processed
f=x-final

URLPATH=https://storage.googleapis.com/paws/pawsx/"$f.tar.gz"

mkdir -p $DATA_RAW

# Download data
wget -c $URLPATH -P $DATA_RAW

# unzip data
tar -zxvf $DATA_RAW/"$f.tar.gz" --directory $DATA_RAW