#!/bin/bash

datasets=(
    ETH3D_SfM_Indoor_test
    ETH3D_SfM_Outdoor_test
    DDAD_val
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
)

for dataset in "${datasets[@]}"
do
    echo "======================================"
    echo "Running dataset: ${dataset}"
    echo "======================================"

    python test.py \
        gpus=[5] \
        name=BP_KITTI_${dataset} \
        ++chpt=BP_KITTI \
        net=PMP \
        num_workers=0 \
        data=UNI \
        data.testset.mode=test \
        data.path=../datasets/uniformat_release/${dataset} \
        test_batch_size=1 \
        metric=MetricALL \
        ++save=true \
        2>&1 | tee "results/${dataset}.log"

done

echo "All tests finished."