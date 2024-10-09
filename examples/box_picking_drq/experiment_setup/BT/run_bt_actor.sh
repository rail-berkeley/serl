export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/BT/bt_policy.py "$@" \
    --env box_picking_camera_env_eval \
    --exp_name=bt_drq_policy \
    --max_traj_length 100 \
    --eval_n_trajs 30 \
