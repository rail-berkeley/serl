export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python bc_policy.py "$@" \
    --env box_picking_basic_env \
    --exp_name=bc_drq_policy \
    --seed 42 \
    --batch_size 256 \
    --eval_checkpoint_step 50000 \
#    --debug # wandb is disabled when debug
