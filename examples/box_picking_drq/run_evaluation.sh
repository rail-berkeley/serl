export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_tests \
    --exp_name=drq_evaluation \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/checkpoints voxnet only pure 32 16 8 noFT (else all) 0816-17:14"\
    --eval_checkpoint_step 12500 \
    --eval_n_trajs 20 \
    \
    --encoder_type voxnet \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --enable_obs_rotation_wrapper \
    --debug
