export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name=drq_robotiq_policy \
    --max_traj_length 100 \
    --camera_mode rgb \
    --seed 0 \
    --max_steps 10000 \
    --random_steps 0 \
    --training_starts 0 \
    --utd_ratio 4 \
    --batch_size 128 \
    --eval_period 1000 \
    --encoder_type resnet-pretrained \
#    --debug