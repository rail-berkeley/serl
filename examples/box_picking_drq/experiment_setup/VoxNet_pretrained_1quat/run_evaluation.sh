export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_tests \
    --exp_name="Voxnet pq temp ens" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/VoxNet_pretrained_1quat/checkpoints voxnet pretrained 1quat 0823-10:45"\
    --eval_checkpoint_step 11000 \
    --eval_n_trajs 50 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --enable_obs_rotation_wrapper \
    --enable_temporal_ensemble_sampling \
    --debug
