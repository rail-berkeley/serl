export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env \
    --max_traj_length 100 \
    --exp_name=box_picking \
    --camera_mode pointcloud \
    --seed 1 \
    --max_steps 20000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 1000 \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --enable_obs_rotation_wrapper \
#    --debug
