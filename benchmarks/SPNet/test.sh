# python test_uni.py \
#     --model_type="Large" --max_depth=1 \
#     --data_dir ../datasets/uniformat_release/iBims_test_2150


#!/bin/bash

ROOT=../datasets/uniformat_release
CKPT=Large

mkdir -p results

datasets=(
    ETH3D_SfM_Indoor_test
    ETH3D_SfM_Outdoor_test
    KITTIDC_test_LiDAR_64
    KITTIDC_test_LiDAR_32
    KITTIDC_test_LiDAR_16
    KITTIDC_test_LiDAR_8
    VOID_sample1500
    VOID_sample500
    VOID_sample150
    NYU_test_500
    NYU_test_200
    NYU_test_100
    NYU_test_50
    NYU_test_5
    DDAD_val
)

for data_name in "${datasets[@]}"
do
    echo "Running ${data_name} ..."

    CUDA_VISIBLE_DEVICES=0 python test_uni.py \
        --model_type="${CKPT}" \
        --max_depth=100.0 \
        --data_dir="${ROOT}/${data_name}" \
        --save_dir="results/${data_name}" \
        2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done
