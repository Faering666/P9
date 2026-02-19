#!/bin/bash

python src/tl-pipeline.py \
	--stage1_data_path data/East-Kaza \
	--stage1_data_type Kazakhstan \
	--stage1_epochs 300 \
	--stage1_lr 4e-4 \
	--stage2_epochs 0 \
	--stage3_epochs 0
