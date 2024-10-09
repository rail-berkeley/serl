export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env \
    --exp_name="SAC no images" \
    --camera_mode none \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 20000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --eval_period 0 \
    \
    --encoder_type none \
    --state_mask all \
#    --debug
