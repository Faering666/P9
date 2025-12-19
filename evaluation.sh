#!/bin/bash

# Perform predictions
prediction_path="results/"
truth_path="./data/Sri-Lanka-Aligned/"
data_type="Sri-Lanka"
model="./checkpoints/hyggestue-19-12-2025/stage1_best_final.pth"
result_path="my_results.json"

python ./src/eval.py \
 --model $model \
 --data_path $truth_path \
 --data_type $data_type \
 --save_path $prediction_path

# Evaluate predictions
python ./src/run_validation.py \
 --pred_path $prediction_path \
 --truth_path $truth_path \
 --type $data_type \
 --result_path $result_path
