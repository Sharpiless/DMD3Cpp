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
    CUDA_VISIBLE_DEVICES=0 python test.py \
      --datapath ../datas/${data_name} \
      --dataset Uni --model raft-stereo \
      --loadmodel kitti-raftstereo.tar --maxdisp 192 \
      --outdir results/ --wsize 5 \
      --guideperc 1 --blending 1 \
      --interpolate --filling \
      --leftpadding --filterlidar \
      --maskocc --iscale 1 \
      --vis_dir results/${data_name} \
      2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done

echo "All tests finished."

