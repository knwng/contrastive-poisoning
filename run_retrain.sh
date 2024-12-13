#!/bin/bash

set -euxo pipefail

# CUDA_VISIBLE_DEVICES=0
# OMP_NUM_THREADS=8

# origin
# python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --dataset cifar10 \
#     --arch resnet18 \
#     --cl_alg [SimCLR/MoCov2/BYOL] \
#     [--classwise or --samplewise] \
#     --delta_weight $[8. / 255] \
#     --folder_name eval_poisons \
#     --epochs 1000 \
#     --eval_freq 100 \
#     --pretrained_delta pretrained_poisons/cifar10_res18_simclr_cps.pth \
#     --sas_subset_indices ../sas-data-efficient-contrastive-learning/final_subsets/cifar10-cl-core-idx.pkl

ALGORITHM="BYOL"
ALGO_IN_SMALLCASE="byol"
SUBSET_FRACTION=0.6
EPOCHS=100
EVAL_FREQ=50
DATASET="/data1/cifar10-${ALGO_IN_SMALLCASE}-poisoned-cps/"
PRETRAINED_DELTA="pretrained_poisons/cifar10_res18_${ALGO_IN_SMALLCASE}_cps.pth"
# SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/final_subsets/cifar10-sas-core-selected-0.8-idx.pkl"
# SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/cifar10-simclr-cps-poisoned-0.2-sas-indices.pkl"
SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/cifar10-${ALGO_IN_SMALLCASE}-cps-poisoned-${SUBSET_FRACTION}-sas-indices.pkl"

# printf "\n\n\n\n----------SAS----------"
# python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --dataset cifar10 \
#     --arch resnet18 \
#     --cl_alg "${ALGORITHM}" \
#     --samplewise \
#     --folder_name eval_poisons \
#     --epochs "${EPOCHS}" \
#     --eval_freq "${EVAL_FREQ}" \
#     --pretrained_delta "${PRETRAINED_DELTA}" \
#     --sas_subset_indices "${SAS_SUBSET_INDICES}"

printf "\n\n\n\n----------Random----------"
# SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/final_subsets/cifar10-rand-selected-${SUBSET_FRACTION}-idx.pkl"
SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/cifar10-rand-selected-${SUBSET_FRACTION}-idx-2.pkl"

python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --dataset cifar10 \
    --arch resnet18 \
    --cl_alg "${ALGORITHM}" \
    --samplewise \
    --folder_name eval_poisons \
    --epochs "${EPOCHS}" \
    --eval_freq "${EVAL_FREQ}" \
    --pretrained_delta "${PRETRAINED_DELTA}" \
    --sas_subset_indices "${SAS_SUBSET_INDICES}"

# printf "\n\n\n\n----------Baseline----------"
# python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --dataset cifar10 \
#     --arch resnet18 \
#     --cl_alg "${ALGORITHM}" \
#     --samplewise \
#     --folder_name eval_poisons \
#     --epochs "${EPOCHS}" \
#     --eval_freq "${EVAL_FREQ}" \
#     --pretrained_delta "${PRETRAINED_DELTA}"


printf "\n\n\n\n----------Random 3----------"
SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/cifar10-rand-selected-${SUBSET_FRACTION}-idx-3.pkl"

python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --dataset cifar10 \
    --arch resnet18 \
    --cl_alg "${ALGORITHM}" \
    --samplewise \
    --folder_name eval_poisons \
    --epochs "${EPOCHS}" \
    --eval_freq "${EVAL_FREQ}" \
    --pretrained_delta "${PRETRAINED_DELTA}" \
    --sas_subset_indices "${SAS_SUBSET_INDICES}"
