export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_eval \
    --exp_name="Voxnet Evaluation unseen" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/VoxNet/checkpoints Voxnet 0821-16:06"\
    --eval_checkpoint_step 11000 \
    --eval_n_trajs 30 \
    \
    --encoder_type voxnet \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --debug
