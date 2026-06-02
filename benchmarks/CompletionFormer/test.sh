mkdir -p results

datasets=(
    DDAD_val
    # KITTIDC_test_LiDAR_64
    # KITTIDC_test_LiDAR_32
    # KITTIDC_test_LiDAR_16
    # KITTIDC_test_LiDAR_8
    # ETH3D_SfM_Indoor_test
    # ETH3D_SfM_Outdoor_test
    # NYU_test_500
    # NYU_test_200
    # NYU_test_100
    # NYU_test_50
    # VOID_sample1500
    # VOID_sample500
    # VOID_sample150
)

for data_name in "${datasets[@]}"
do
    echo "Running ${data_name} ..."

    python main.py \
        --dir_data ../../datasets/uniformat_release/${data_name} \
        --data_name UNI \
        --gpus 5 --max_depth 90.0 --save_image \
        --test_only --pretrain ../KITTIDC_L1L2.pt --save results/${data_name} \
        2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done

echo "All tests finished."