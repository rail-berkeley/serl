export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name=drq_robotiq_policy \
    --max_traj_length 300 \
    --seed 0 \
    --max_steps 10000 \
    --random_steps 100 \
    --training_starts 100 \
    --utd_ratio 4 \
    --batch_size 512 \
    --eval_period 1000 \
    --encoder_type resnet-pretrained \
