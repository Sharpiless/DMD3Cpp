# CUDA_VISIBLE_DEVICES=3 python test_uni.py \
#     --data_dir ../datasets/uniformat_release/iBims_test_2150 \
#     --output_dir results_marigold_dc/iBims_test_2150 \
#     --save_results

#!/bin/bash

ROOT=../datasets/uniformat_release
CKPT=L_DA_DepthPro.pth

mkdir -p results

datasets=(
    # ETH3D_SfM_Indoor_test
    # ETH3D_SfM_Outdoor_test
    # KITTIDC_test_LiDAR_64
    # KITTIDC_test_LiDAR_8
    # VOID_sample1500
    # VOID_sample500
    # VOID_sample150
    # NYU_test_500
    # NYU_test_50
    # NYU_test_500
    # NYU_test_200
    # NYU_test_100
    # NYU_test_50
    DDAD_val
)

for data_name in "${datasets[@]}"
do
    echo "Running ${data_name} ..."

    CUDA_VISIBLE_DEVICES=5 python test_uni.py \
        --data_dir "${ROOT}/${data_name}" \
        2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done
