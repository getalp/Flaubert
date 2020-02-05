#!/usr/bin/env bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

set -e

# data paths
DATA_DIR=$1
DATA_RAW=$DATA_DIR/raw

URLPATH=https://zenodo.org/record/3251672/files
f=cls-acl10-processed
fu=cls-acl10-unprocessed

mkdir -p $DATA_RAW

## Download data
wget -c $URLPATH/$fu.tar.gz -P $DATA_RAW

# unzip data
tar -zxvf $DATA_RAW/$fu.tar.gz --directory $DATA_RAW