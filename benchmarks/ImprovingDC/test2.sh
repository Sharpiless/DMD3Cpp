# export LD_LIBRARY_PATH=/data7/liangyingping/cuda-11.3/lib64:$LD_LIBRARY_PATH
# export PATH=/data7/liangyingping/cuda-11.3/bin:$PATH
# export CUDA_HOME=/data7/liangyingping/cuda-11.3
# # conda activate bpnet

# python val.py -c val_uni.yml -d ../datas/DDAD_val 2>&1 | tee "results/DDAD_val.log"

datasets=(
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
    KITTIDC_test_LiDAR_64
    KITTIDC_test_LiDAR_32
    KITTIDC_test_LiDAR_16
    KITTIDC_test_LiDAR_8
)
mkdir -p results

for data_name in "${datasets[@]}"
do
    echo "Running ${data_name} ..."
    python val2.py -c val_uni.yml -d ../datas/${data_name} \
        2>&1 | tee "results/${data_name}.log"

    echo "Finished ${data_name}"
done

echo "All tests finished."