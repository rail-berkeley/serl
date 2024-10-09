export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy.py "$@" \
    --actor \
    --env box_picking_basic_env \
    --exp_name=sac_drq_policy_evaluation \
    --eval_checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_sac/checkpoints 0411-16:46"\
    --eval_checkpoint_step 100000 \
    --eval_n_trajs 10 \
    --debug
