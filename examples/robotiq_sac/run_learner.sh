export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_robotiq.py "$@" \
    --learner \
    --env robotiq-grip-v1 \
    --exp_name=sac_robotiq_policy \
    --seed 0 \
    --random_steps 600 \
    --training_starts 600 \
    --utd_ratio 8 \
    --batch_size 1024 \
    --eval_period 2000 \
    --max_steps 10000 \
    --preload_rlds_path "/home/nico/real-world-rl/serl/examples/robotiq_sac/rlds" \
#    --demo_paths "/home/nico/real-world-rl/serl/examples/robotiq_bc/robotiq_test_20_demos_2024-03-26_12-23-50.pkl" \
#    --debug
