#!/bin/bash
# Copyright 2019 Hang Le (hangtp.le@gmail.com)

# Argument 1 is number of jobs, argument 2 is the config file
if [ $# -gt 1 ]
then
    Njobs=$1
    CONFIG=$2
    echo "Number of jobs (excluding the first one): "$Njobs
    echo "Configuration file: "$CONFIG
else
    echo "Require 2 arguments!!!!!!! Exit."
    exit
fi

# Read the configuration variables
# Each training should have a difference config
source $CONFIG

# # first job - no dependencies
# j0=$(sbatch train.slurm $CONFIG)
# # Format of j0: "Submitted job <jobID>". Here we get the true jobID
# j0_id=$(echo $j0 | sed 's/[^0-9]*//g')
# echo "ID of the first job: $j0_id"

# Resubmitted due to disruption 
j_resumed=$3
RESUME_CHECKPOINT_j0=$OUTPUTPATH/$EXPNAME/$j_resumed/checkpoint.pth
j0=$(sbatch train.slurm $CONFIG $RESUME_CHECKPOINT_j0)
j0_id=$(echo $j0 | sed 's/[^0-9]*//g')
echo "ID of the first job - resumed job: $j0_id"
echo "This job will resume training from $RESUME_CHECKPOINT_j0."

# add first job to the list of jobs
jIDs+=($j0_id)

# for loop: submit Njobs: where job i + 1 is dependent on job i.
# and job i + 1 resume from the checkpoint of job i
for i in $(seq 1 $Njobs); do
    # Submit job i+1 (i.e. new_job) with dependency ('afterok:') on job i
    RESUME_CHECKPOINT=$OUTPUTPATH/$EXPNAME/${jIDs[ $i - 1 ]}/checkpoint.pth
    new_job=$(sbatch --dependency=afterok:${jIDs[ $i - 1 ]} train.slurm $CONFIG $RESUME_CHECKPOINT)
    new_job_id=$(echo $new_job | sed 's/[^0-9]*//g')
    echo "Submitted Job ID $new_job_id that will be executed once Job ID ${jIDs[ $i - 1 ]} has completed with success."
    echo "This job will resume training from $RESUME_CHECKPOINT."
    jIDs+=($new_job_id)
    echo "List of jobs that have been submitted: ${jIDs[@]}"
done