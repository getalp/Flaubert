#! bin/bash

CUDA_VISIBLE_DEVICES=0 python run_flue.py \
                --model_type flaubert \
                --model_name_or_path ~/pretrained_models/Flaubert/flaubert_base_cased \
                --task_name SST-2 \
                --do_train \
                --do_eval \
                --data_dir ~/Data/Finetune-BERT-FR/cls/data_processed_huggingface/books \
                --max_seq_length 512 \
                --per_gpu_train_batch_size 8 \
                --learning_rate 5e-6 \
                --num_train_epochs 30 \
                --output_dir ~/Experiments/Finetune/Flaubert/cased \
                --save_steps 20000 \
                |& tee output.log