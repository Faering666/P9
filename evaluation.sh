# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

#!/bin/bash

prediction_path="results/"
truth_path="./data/Sri-Lanka-Aligned/"
data_type="Sri-Lanka"
model="./checkpoints/hyggestue-19-12-2025/stage1_best_final.pth"
result_path="my_results.json"

# Perform predictions
# echo "=== Beginning predictions ==="
# python ./src/eval.py \
#  --model $model \
#  --data_path $truth_path \
#  --data_type $data_type \
#  --save_path $prediction_path

# Evaluate predictions
echo "=== Beginning evaluation  ==="
python ./src/run_validation.py \
 --pred_path $prediction_path \
 --truth_path $truth_path \
 --type $data_type \
 --result_path $result_path