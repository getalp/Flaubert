#! bin/bash
# Hang Le (hangtp.le@gmail.com)
# Modified from
# https://github.com/facebookresearch/XLM
# Original copyright is appended below.
# 
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

# Script to split data to 3 datasets: train, valid and test
# Syntax to run this script:
## ./split_train_val_test.sh $DATA_PATH

lg=fr

# Specify parameters to run the script
DATA_PATH=$1 # path to where you save the file to be split
dname=`dirname $DATA_PATH`

# Split into train / valid / test
echo "***** Split into train / validation / test datasets *****"

split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
};

    local train_percent=$2 # percent of data to split for train
    local val_percent=$3 # percent of data to split for validation (also percent for test)

    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=`echo "($train_percent * $NLINES)/1" | bc`;
    num_val=`echo "($val_percent * $NLINES)/1" | bc`;
    NVAL=$((NTRAIN + num_val));
    num_test=$((NLINES - NVAL));

    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN                 > $4;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -$num_val  > $5;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -$num_test               > $6;
}
split_data $DATA_PATH 0.99 0.005 $dname/$lg.train $dname/$lg.valid $dname/$lg.test