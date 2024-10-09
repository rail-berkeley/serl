export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/nico/real-world-rl/serl/examples/box_picking_bc/bc_policy.py "$@" \
    --env box_picking_camera_env \
    --exp_name=bc_drq_policy \
    --seed 1 \
    --batch_size 2048 \
    --max_steps 25000 \
    --demo_paths /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/box_picking_20_demos_2024-08-20_state_only.pkl \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/BC/checkpoints \
    --checkpoint_period 1000 \
    --eval_checkpoint_step 0 \
#    --debug # wandb is disabled when debug
