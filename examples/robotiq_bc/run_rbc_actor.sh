export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python bc_policy_robotiq.py "$@" \
    --env robotiq_basic_env \
    --exp_name=bc_robotiq_policy \
    --seed 42 \
    --batch_size 256 \
    --eval_checkpoint_step 50000 \
#    --debug # wandb is disabled when debug
