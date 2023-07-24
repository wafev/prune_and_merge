#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONWARNINGS="ignore"

#python finetune.py --eval --model tmvit_base_patch16_224  --batch-size 64 --num_workers 32 --data-path ./data/imagenet \
#--resume /home/maojunzhu/pycharmprojects/tm-vit/experiments/b16_augreg_finetune/best.pth \
#--output_dir ./experiments/B16-augreg_01_06 --speed-only

#python -m torch.distributed.launch --master_port=3002 --nproc_per_node=4 --use_env \
python finetune.py --model tmvit_base_patch16_224  \
--resume /home/maojunzhu/pycharmprojects/tm-vit/experiments/B16-augreg/augreg.pth --data-path /data/imagenet \
--prune_mode attn --prune_rate 0.1 --keep_rate 0.60  --final_finetune 50 --warmup-epochs 0 \
--batch-size 256 --num_workers 16 --lr 1e-4 --weight-decay 0.05 --distillation-alpha 1 --distillation-tau 20 \
--output_dir ./experiments/b16_augreg_01_06 --drop-path 0.1 \
--teacher-path /home/maojunzhu/pycharmprojects/tm-vit/experiments/b16_augreg_finetune/best.pth \
--teacher-model tmvit_base_patch16_224 --distillation-type soft --mixup 0 --reprob 0 --cutmix 0 --prune --prune_iter 1000 # \


#python -m torch.distributed.launch --master_port=3001 --nproc_per_node=2 --use_env \
#python finetune.py --model tmvit_small_patch16_224 --freeze_matrix \
#--resume /home/maojunzhu/pycharmprojects/tm-vit/experiments/s16-augreg-01-60-stdice/best.pth --data-path /data/imagenet \
#--prune_mode attn --prune_rate 0 --keep_rate 0.6 --finetune_nums 0 --final_finetune 10 --warmup-epochs 0 \
#--batch-size 512 --num_workers 32 --lr 1e-6 --weight-decay 0.001 --distillation-alpha 1 --distillation-tau 20 \
#--output_dir ./experiments/s16-augreg-01-60-stdice  \
#--teacher-path /home/maojunzhu/pycharmprojects/tm-vit/experiments/s16-augreg-finetune/checkpoint.pth \
#--teacher-model tmvit_small_patch16_224 --distillation-type soft --mixup 0 --reprob 0 --cutmix 0 \
#--prune --prune_iter 250
#