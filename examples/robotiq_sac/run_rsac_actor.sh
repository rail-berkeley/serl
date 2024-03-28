export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python /home/nico/real-world-rl/serl/examples/robotiq_bc/bc_policy_robotiq.py "$@" \
    --env robotiq-grip-v1 \
    --exp_name=sav_robotiq_actor \
    --seed 42 \
    --batch_size 256 \
    --eval_checkpoint_step 100000 \
#    --debug # wandb is disabled when debug
