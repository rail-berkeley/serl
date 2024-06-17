export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --env PandaPickCube-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --debug # wandb is disabled when debug
