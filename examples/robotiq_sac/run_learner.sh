export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_robotiq.py "$@" \
    --learner \
    --env robotiq-grip-v1 \
    --exp_name=sac_robotiq_policy \
    --seed 0 \
    --random_steps 400 \
    --training_starts 400 \
    --utd_ratio 8 \
    --batch_size 1024 \
    --eval_period 1000 \
    --max_steps 100000 \
    --reward_scale 1 \
    --demo_paths "/home/nico/real-world-rl/serl/examples/robotiq_bc/robotiq_test_20_demos_apr9_action_cost.pkl" \
#    --preload_rlds_path "/home/nico/real-world-rl/serl/examples/robotiq_sac/rlds" \
#    --debug
