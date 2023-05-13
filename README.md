# SC-prompt
## Introduction
This repository contains the code for the paper "Few-shot Text-to-SQL Translation using Structure and Content Prompt Learning". In this paper, we propose SC-Prompt, a novel divide-and-conquer strategy for effectively supporting Text-to-SQL translation in the few-shot scenario. 

## Setup
```sh
git clone git@github.com:ruc-datalab/SC-prompt.git
cd SC-prompt
mkdir -p -m 777 experimental_outputs
mkdir -p -m 777 transformers_cache
cd experimental_outputs
mkdir -p -m 777 spider
mkdir -p -m 777 cosql
mkdir -p -m 777 geoquery
cd ..
```

## Dataset Download

- [Spider](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0): Put it under `src/datasets/spider`.
- [Cosql](https://drive.google.com/uc?export=download&id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP): Put it under `src/datasets/cosql`.
- [Geoquery](https://drive.google.com/file/d/1hP4gpExG1EJCN3a1vOyK4XR4mTSFi7Q1/view?usp=share_link): Put it under `src/datasets/geoquery`.

## Code Structure

```sh
|-- experimental_outputs # save the fine-tuned models and evaluation results
|-- scripts # the train/inference script
|-- src
    |-- datasets # the class to preprocess the dataset 
    |-- metrics # the class to evaluate the prediction results
    |-- utils # main code
    |-- run.py # the class to train/inference the few-shot text-to-sql model
```

## Environment
Our constrained decoding method is based on the parser provided by [Picard](https://arxiv.org/abs/2109.05093). Please use the Docker image provided by the official [repository](https://github.com/ServiceNow/picard) to build the container.

```sh
docker run -itd --gpus '"device=<your_available_gpu_ids>"' --rm --user 13011:13011 --mount type=bind,source=<your_base_dir>/transformers_cache,target=/transformers_cache --mount type=bind,source=<your_base_dir>/scripts,target=/app/scripts --mount type=bind,source=<your_base_dir>/experimental_outputs,target=/app/experimental_outputs --mount type=bind,source=<your_base_dir>/src,target=/app/src tscholak/text-to-sql-eval:6a252386bed6d4233f0f13f4562d8ae8608e7445
```
You should set `<your_available_gpu_ids>` and `<your_base_dir>`.

## Quick Inference

Download the fine-tuned model and put it under the corresponding folder.

| Dataset | #Train | Model | Folder |
|-------|--------|--------|---------|---------|-----------------------|----------------|
| Spider | 0.05 (350) | [link](https://drive.google.com/drive/folders/1b-16LFsnVMC5U2JxRew9nKtdOIhVr46j?usp=share_link) | experimental_outputs/spider/ |
| Spider | 0.1 (700) | [link](https://drive.google.com/drive/folders/16qcI-zcahpB-Y6BUyizLmt3-EMP8_sM7?usp=share_link) | experimental_outputs/spider/ |
| CoSQL | 0.05 (475) | [link](https://drive.google.com/drive/folders/1DxNdW5oBMQgYm7GE_VfvT9lFrJLcCpLs?usp=share_link) | experimental_outputs/cosql/ |
| CoSQL | 0.1 (950) | [link](https://drive.google.com/drive/folders/1MhbsPsyhD0RTVYFJ7jiqy8zxxUo2_4kp?usp=share_link) | experimental_outputs/cosql/ |
| Geoquery | 1. (536) | [link](https://drive.google.com/drive/folders/1Z-akKlTFhiNGdT23kmpU8VFQ3L5XvOgD?usp=share_link) | experimental_outputs/geoquery/ |

Use the scripts to inference.
```sh
# Inference on spider
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_spider_scprompt.sh 0.1
# Inference on cosql
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_cosql_scprompt.sh 0.1
# Inference on geoquery
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_geoquery_scprompt.sh 1.
```
- The second argument refers to the proportion of using the official training set.

## Train from scrach
```sh
# Train on spider
CUDA_VISIBLE_DEVICES=0 bash scripts/train_spider_scprompt.sh 0.1
# Train on cosql
CUDA_VISIBLE_DEVICES=0 bash scripts/train_cosql_scprompt.sh 0.1
# Train on geoquery
CUDA_VISIBLE_DEVICES=0 bash scripts/train_geoquery_scprompt.sh 1.
```
- The second argument refers to the proportion of using the official training set.

The best model will be automatically saved at `experimental_outputs/`. Please note that training does not use the fine-grained constrained decoding strategy, which is only necessary for evaluation. Please refer to `Quick Inference`to evaluate the fine-tuned model.

