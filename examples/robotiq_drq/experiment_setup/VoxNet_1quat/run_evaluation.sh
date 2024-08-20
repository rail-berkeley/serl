export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/robotiq_drq/drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env_tests \
    --exp_name="voxnet 1quat" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/experiment_setup/VoxNet_1quat/checkpoints voxnet only pure 32 16 8 noFT (else all) 0820-12:50"\
    --eval_checkpoint_step 10000 \
    --eval_n_trajs 10 \
    \
    --encoder_type voxnet \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --enable_obs_rotation_wrapper \
    --debug
