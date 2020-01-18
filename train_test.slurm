#!/bin/bash
# Copyright 2019 Hang Le (hangtp.le@gmail.com)

#SBATCH --job-name=test_scaling    # nom du job
#SBATCH --partition=gpu_p1                     # demande d'allocation sur la partition GPU
#SBATCH --exclusive
#SBATCH --exclude=r8i2n7
#SBATCH --ntasks=2                 # nombre de tâche MPI
#SBATCH --ntasks-per-node=2         # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:2                # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10            # nombre de coeurs à réserver par tâche
#SBATCH --hint=nomultithread              
#SBATCH --time=00:05:00              # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=test_scaling_%j.out # nom du fichier de sortie
#SBATCH --error=test_scaling_%j.stderr  # nom du fichier d'erreur (ici commun avec la sortie)


# chargement des modules
module purge
# module load pytorch-gpu/py3/1.1
# module load pytorch-gpu/py3/1.3.1
module load pytorch-gpu/py3/1.3.1+nccl-2.5.6
 
# echo des commandes lancées
set -x

# Argument 1 is the configuration file
if [ $# -gt 0 ]
then
    CONFIG=$1
    source $CONFIG
    echo "Experiment name: $EXPNAME"
else
    echo "First argument (configuration file) must be provided!"
    exit 1
fi

# Argument 2 is the checkpoint file
if [ $# -gt 1 ]
then
    RESUME_CHECKPOINT=$2
    echo "Resume training from this checkpoint: "$RESUME_CHECKPOINT
else
    echo "No checkpoint provided. Training from scratch!"
    RESUME_CHECKPOINT=""
fi

mkdir -p $OUTPUTPATH/$EXPNAME

cd $RUNPATH
echo "Current working directory: $PWD."
echo "LayerDrop - Prenorm - LayerNorm epsilon: $layerdrop - $pre_norm - $layer_norm_eps"

# Check memory available
cat /sys/fs/cgroup/memory/`grep memory /proc/self/cgroup | cut -d: -f 3`/memory.limit_in_bytes

export NCCL_DEBUG=INFO

echo "Start training ..."

srun python train.py \
    --exp_name $EXPNAME \
    --dump_path $OUTPUTPATH \
    --data_path $DATAPATH \
    --reload_checkpoint "$RESUME_CHECKPOINT" \
    --time_limit $TIMELIMIT \
    --amp $amp \
    --lgs 'fr' \
    --clm_steps '' \
    --mlm_steps 'fr' \
    --emb_dim $EMB_DIM \
    --n_layers $LAYERS \
    --n_heads $HEADS \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --batch_size $BATCH_SIZE \
    --bptt $MAX_LEN \
    --optimizer $optimizer \
    --epoch_size $EPOCH_SIZE \
    --max_epoch 100000 \
    --validation_metrics _valid_fr_mlm_ppl \
    --stopping_criterion _valid_fr_mlm_ppl,20 \
    --fp16 $fp16 \
    --accumulate_gradients $ACCUM_GRAD \
    --word_mask_keep_rand '0.8,0.1,0.1' \
    --word_pred '0.15' \
    --layerdrop $layerdrop \
    --pre-norm $pre_norm \
    --layer-norm-eps $layer_norm_eps \
    --use_apex $use_apex \
    --master_port 20000 \
    |& tee $OUTPUTPATH/$EXPNAME/output.log