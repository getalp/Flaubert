#!/bin/bash

# cancel all jobs between jobID1, jobID2 (including them)

# Argument 1: jobID1, argument 2: jobID2
if [ $# -gt 1 ]
then
    id1=$1
    id2=$2
    echo "first job: "$id1
    echo "last job: "$id2
else
    echo "Require 2 arguments!!!!!!! Exit."
    exit
fi

for i in $(seq $id1 $id2); do 
    scancel $i;
done