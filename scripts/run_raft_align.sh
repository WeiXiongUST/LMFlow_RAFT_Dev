#!/bin/bash

# The total iteration of raft
raft_num_iteration=6


base_dir="/home/xiongwei/raft/LMFlow_RAFT_Dev/output_models/raft_test"
mkdir $base_dir
sft_model="/home/xiongwei/LMFlow/output_models/sft_llama_7b_2e-5_1epoch"
#sft_model="gpt2"
reward_model="/home/xiongwei/LMFlow/output_models/openllama_3b_rm_2sft_full_train_5e-6_1epoch_4x8bs_raw_dataset"

x=0
y=1
model_dir="${base_dir}/model${x}"
mkdir ${model_dir}
tmp_model_dir="${base_dir}/model${y}"

mkdir $tmp_model_dir
mkdir ${model_dir}/infer_set
mkdir ${model_dir}/filtered_set
mkdir ${tmp_model_dir}/infer_set
mkdir ${tmp_model_dir}/filtered_set

#CUDA_VISIBLE_DEVICES="4,5,6,7" ./scripts/infer_get_samples.sh ${sft_model} 0 ${model_dir}/infer_set
#CUDA_VISIBLE_DEVICES="4,5,6,7" ./scripts/infer_get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${base_dir} ${reward_model}
CUDA_VISIBLE_DEVICES="4,5,6,7" ./scripts/finetune.sh ${sft_model} $tmp_model_dir ${model_dir}/filtered_set


old_model_dir=$tmp_model_dir 

for (( i=2; i<=$raft_num_iteration; i++ )); do
  model_dir="${base_dir}/model${i}"
  mkdir $model_dir
  mkdir ${model_dir}/infer_set
  mkdir ${model_dir}/filtered_set
  
  CUDA_VISIBLE_DEVICES="4,5,6,7" ./scripts/infer_get_samples.sh $old_model_dir $((i - 1)) ${old_model_dir}/infer_set
  CUDA_VISIBLE_DEVICES="4,5,6,7" ./scripts/infer_get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${base_dir} ${reward_model}
  CUDA_VISIBLE_DEVICES="4,5,6,7"  ./scripts/finetune.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set

  old_model_dir=$model_dir
done

# for (( i=1; i<=$count; i++ )); do
#   model_dir="${base_dir}/model${i}"
#   mkdir $model_dir/eval_set
#   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ./scripts_for_uncertainty_study/eval_get_samples.sh $model_dir
# done

# for (( i=1; i<=$count; i++ )); do
#   model_dir="${base_dir}/model${i}"
#   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ./scripts_for_uncertainty_study/infer_get_rewards_with_gold.sh $model_dir/eval_set /home/xiongwei/LMFlow/output_models/test_infer ${base_dir}
# done