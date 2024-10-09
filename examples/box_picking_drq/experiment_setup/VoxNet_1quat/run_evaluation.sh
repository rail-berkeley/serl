export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_eval \
    --exp_name="Voxnet 1quat Evaluation" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/VoxNet_1quat/checkpoints voxnet 1quat 0822-17:54"\
    --eval_checkpoint_step 18000 \
    --eval_n_trajs 30 \
    \
    --encoder_type voxnet \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --enable_obs_rotation_wrapper \
#    --debug
