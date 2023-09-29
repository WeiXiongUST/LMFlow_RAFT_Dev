#!/bin/bash

# The total iteration of raft
raft_num_iteration=6


base_dir="/home/usr/raft/LMFlow_RAFT_Dev/output_models/raft_test"
mkdir $base_dir
# You should edit the sft model dir accordingly
sft_model=gpt2
reward_model="weqweasdas/hh_rlhf_rm_open_llama_3b"

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

./scripts/infer_get_samples.sh ${sft_model} 0 ${model_dir}/infer_set
./scripts/infer_get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${base_dir} ${reward_model}
./scripts/finetune.sh ${sft_model} $tmp_model_dir ${model_dir}/filtered_set


old_model_dir=$tmp_model_dir 

for (( i=2; i<=$raft_num_iteration; i++ )); do
  model_dir="${base_dir}/model${i}"
  mkdir $model_dir
  mkdir ${model_dir}/infer_set
  mkdir ${model_dir}/filtered_set
  
  ./scripts/infer_get_samples.sh $old_model_dir $((i - 1)) ${old_model_dir}/infer_set
  ./scripts/infer_get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${base_dir} ${reward_model}
  ./scripts/finetune.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set

  old_model_dir=$model_dir
done
