export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name="Depth image small encoder" \
    --camera_mode depth \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 50000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/Depth\ Image/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/box_picking_20_demos_08-24_15-40-34_depth_20cm.pkl \
    \
    --encoder_type small \
    --state_mask all \
#    --debug
