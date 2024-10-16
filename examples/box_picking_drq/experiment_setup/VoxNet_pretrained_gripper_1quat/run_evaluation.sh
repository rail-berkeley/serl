export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_tests \
    --exp_name="voxnet [pqg] temp ens" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/VoxNet_pretrained_gripper_1quat/checkpoints voxnet pretrained gripper_only 1quat 0829-14:08"\
    --eval_checkpoint_step 11000 \
    --eval_n_trajs 50 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask gripper \
    --encoder_bottleneck_dim 128 \
    --enable_obs_rotation_wrapper \
    --enable_temporal_ensemble_sampling \
    --debug
