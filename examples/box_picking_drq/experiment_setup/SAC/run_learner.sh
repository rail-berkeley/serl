export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name="SAC no images" \
    --camera_mode none \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 50000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --checkpoint_period 2500 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/SAC/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/box_picking_20_demos_2024-08-20_rgb_depth.pkl \
    \
    --encoder_type none \
    --state_mask all \
#    --debug
