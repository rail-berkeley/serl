export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python /home/nico/real-world-rl/serl/examples/box_picking_bc/bc_policy.py "$@" \
    --env box_picking_camera_env_eval \
    --exp_name=bc_drq_policy \
    --seed 1 \
    --max_traj_length 100 \
    --batch_size 2048 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/BC/checkpoints \
    --eval_checkpoint_step 5000 \
    --eval_n_trajs 30 \
    --debug # wandb is disabled when debug
