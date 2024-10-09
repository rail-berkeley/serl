export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_eval \
    --exp_name="Depth Image Evaluation" \
    --camera_mode depth \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/Depth Image/checkpoints Depth image small encoder 0827-16:55"\
    --eval_checkpoint_step 22000 \
    --eval_n_trajs 30 \
    \
    --encoder_type small \
    --state_mask all \
#    --debug
