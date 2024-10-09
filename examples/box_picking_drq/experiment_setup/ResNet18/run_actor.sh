export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env \
    --exp_name="ResNet18 feat red 32 128" \
    --camera_mode rgb \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 20000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 96 \
    --eval_period 0 \
    \
    --encoder_type resnet-pretrained-18 \
    --encoder_bottleneck_dim 128 \
    --state_mask all \
    --encoder_kwargs pooling_method \
    --encoder_kwargs feature_reduction \
    --encoder_kwargs num_kp \
    --encoder_kwargs 32 \
#    --debug
