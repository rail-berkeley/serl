export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_robotiq.py "$@" \
    --learner \
    --env robotiq-grip-v1 \
    --exp_name=sac_robotiq_policy \
    --max_traj_length 300 \
    --seed 42 \
    --training_starts 900 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 50000 \
    --reward_scale 1 \
    --demo_paths "/home/nico/real-world-rl/serl/examples/robotiq_sac/robotiq_test_20_demos_apr11_random_boxes.pkl" \
#    --preload_rlds_path "/home/nico/real-world-rl/serl/examples/robotiq_sac/rlds" \
#    --debug
