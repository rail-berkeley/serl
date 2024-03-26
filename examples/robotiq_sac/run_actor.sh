export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python sac_policy_robotiq.py "$@" \
    --actor \
    --env robotiq-grip-v1 \
    --exp_name=sac_robotiq_policy \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 1000 \
    --utd_ratio 8 \
    --batch_size 256 \
    --eval_period 2000
#    --debug
