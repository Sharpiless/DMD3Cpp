#!/bin/bash

ROOT=../datasets/uniformat_release
CKPT=L_DA_DepthPro.pth

mkdir -p results_resize
mkdir -p vis_results

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
    DDAD_val
)

for data_name in "${datasets[@]}"
do
    echo "Running ${data_name} ..."

    CUDA_VISIBLE_DEVICES=7 python test_uni.py \
        --ckpt_path="${CKPT}" \
        --max_depth=150 \
        --data_dir="${ROOT}/${data_name}" \
        --save_results=True \
        --save_dir="vis_results/${data_name}" \
        --cmap=Spectral \
        2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done

echo "All tests finished."