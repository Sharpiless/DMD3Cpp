torchrun --nproc_per_node=8 --master_port 4321 train_distill.py \
    gpus=[0,1,2,3,4,5,6,7] num_workers=4 name=PMP_Residual_Norm_ssil_KITTI \
    net=PMP_Residual_Norm_fast data=KITTI \
    lr=1e-3 train_batch_size=1 test_batch_size=1 loss=MSMSE_ssil \
    sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
    nepoch=30 test_epoch=25 ++net.sbn=true

