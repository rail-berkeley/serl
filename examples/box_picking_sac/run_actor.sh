export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy.py "$@" \
    --actor \
    --env box_picking_basic_env \
    --exp_name=sac_drq_policy \
    --max_traj_length 300 \
    --seed 42 \
    --max_steps 10000 \
    --random_steps 0 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --eval_period 1000 \
    --reward_scale 1 \
#    --debug