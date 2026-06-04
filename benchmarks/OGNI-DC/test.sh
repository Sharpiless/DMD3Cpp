GRU_iters=5
test_augment=0
optim_layer_input_clamp=100.0
depth_activation_format='linear'
depth_downsample_method='min'
pred_confidence_input=1


ckpt=/data6/liangyingping/TPAMI-depth/DMD3Cpp/OMNI-DC/OGNI-DC/checkpoints/KITTI_generalization.pt

mkdir -p results

datasets=(
    KITTIDC_test_LiDAR_64
    KITTIDC_test_LiDAR_32
    KITTIDC_test_LiDAR_16
    KITTIDC_test_LiDAR_8
    ETH3D_SfM_Indoor_test
    ETH3D_SfM_Outdoor_test
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
    python main.py --dir_data ../../datasets/uniformat_release/${data_name} \
        --data_name Uniformat --split_json ../data_json/kitti_dc_test.json \
        --patch_height 240 --patch_width 1216 --lidar_lines 64 \
        --gpus 2 --max_depth 90.0 \
        --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp \
        --depth_activation_format $depth_activation_format \
        --depth_downsample_method $depth_downsample_method --pred_confidence_input $pred_confidence_input \
        --test_only --test_augment $test_augment --pretrain $ckpt \
        --log_dir ../experiments/ \
        --save "results/${data_name}" \
        --save_result_only \
        2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done

echo "All tests finished."