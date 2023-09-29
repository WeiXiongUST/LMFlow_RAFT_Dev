<p align="center" width="100%">
<img src="assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_es.md">Español</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_jp.md">日本語</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_ko.md">한국어</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_hindi.md">हिंदी</a>
    <p>
</h4>

[![Website](https://img.shields.io/badge/Website-Demo-20B2AA.svg)](https://lmflow.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/Discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://s1.ax1x.com/2023/08/06/pPAQTPI.jpg)

This is a sub branch of LMFlow, used to present an implementation of RAFT with separate stages, in the sense that we will implement inference, reward ranking, and finetuning separately, so that as long as you can finetune the model, you might leverage RAFT to align your model.

<p align="center" width="100%">
<img src="assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Quick Start

### Setup

```bash
git clone -b raft_dev https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
./install.sh
```

### Prepare Dataset
We have prepared the HH-RLHF dataset and preprocess it into SFT, RM, and RLHF datasets. 

```bash
cd data
./download.sh hh_rlhf
```


## RLHF Pipeline

### 1 SFT

```bash
./scripts/finetune.sh openlm-research/open_llama_7b ./output_models/sft_open_llama7b ./data/hh_rlhf/sft
```

We skip the reward modeling first and present an overview of the reward modeling later.

### 2 RAFT ALIGNMENT

To run RAFT, we should first modify scripts/run_raft_align.sh including
- The number of total iterations: raft_num_iteration=20;
- The base dir of the experiment used to store all the intermediate models and data: base_dir="./LMFlow_RAFT_Dev/output_models/raft_test"
- The starting checkpoint: sft_model="./output_models/sft_open_llama7b"
- The reward model used for RLHF: reward_model="/home/xiongwei/LMFlow/output_models/openllama_3b_rm_2sft_full_train_5e-6_1epoch_4x8bs_raw_dataset".

Then, we simply run the following script to iteratively calling
- Inference: ./scripts/infer_get_samples.sh A B C
  - parameter A: the model used to collect new data;
  - parameter B: the global iteration id in \{0, 1, ..., raft_num_iteration\}, please do not modify this parameter;
  - parameter C: the dir to save the collected dataset.
- Reward ranking: ./scripts/infer_get_rewards.sh A B C D
  - parameter A: the collected dataset in the inference stage;
  - parameter B: the dir to save the filtered dataset;
  - parameter C: the base dir of the experiment, used to record reward;
  - parameter D: reward model.
- Finetuning: ./scripts/finetune.sh A B C
  - parameter A: the model to be trained;
  - parameter B: the dir to save the trained model;
  - parameter C: the filtered dataset from the reward ranking stage.

```bash
./scripts/run_raft_align.sh
```

#### 2.1 More hyper-parameters



## Support

If you need any help, please submit a Github issue or contact me via wx13@illinois.edu

## License
The code included in this project is licensed under the [Apache 2.0 license](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE).
If you wish to use the codes and models included in this project for commercial purposes, please sign this [document](https://docs.google.com/forms/d/e/1FAIpQLSfJYcci6cbgpIvx_Fh1xDL6pNkzsjGDH1QIcm4cYk88K2tqkw/viewform?usp=pp_url) to obtain authorization.

## Citation
If you find this repository useful, please consider giving ⭐ and citing our [paper](https://arxiv.org/abs/2306.12420):

```
@misc{lmflow,
  author = {Shizhe Diao and Rui Pan and Hanze Dong and KaShun Shum and Jipeng Zhang and Wei Xiong and Tong Zhang},
  title = {LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://optimalscale.github.io/LMFlow/}},
}
```
