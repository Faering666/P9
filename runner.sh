#!/bin/bash
# Base run, with transferlearning on WeedyRice
python src/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 300 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

mkdir checkpoints/basemodel-tl-Weed
mv checkpoints/s* checkpoints/basemodel-tl-Weed

#Using base model, skipping stage2, transferlearning on sri
python src/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_epochs 0 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

mkdir checkpoints/Sri-Lanka-stage3-only
mv checkpoints/s* checkpoints/Sri-Lanka-stage3-only


#Using base model, transferlearning on aligned-sri-lanka
python src/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

 mkdir checkpoints/tl-sri-lanka
 mv checkpoints/s* checkpoints/tl-sri-lanka

#Using base model, skipping stage 2, transferlearning on WeedyRice
python src/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_epochs 0 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

mkdir checkpoints/weed-rice-stage3-only
mv checkpoints/s* checkpoints/weed-rice-stage3-only

#Using stage3 model, transferlearning on Sri-lanka
python src/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_model checkpoints/Sri-Lanka-stage3-only/stage3_best_final.pth \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 0 \
 --stage3_lr 1e-7 

mkdir checkpoints/Sri-lanka-stage2-trained-on-stage3
mv checkpoints/s* checkpoints/Sri-lanka-stage2-trained-on-stage3

#Using stage3 model, transferlearning on WeedyRice
python src/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_model checkpoints/weed-rice-stage3/stage3_best_final.pth \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 0 \
 --stage3_lr 1e-7 

mkdir checkpoints/Weedy-Rice-stage2-trained-on-stage3
mv checkpoints/s* checkpoints/Sri-lanka-stage2-trained-on-stage3