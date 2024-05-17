export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name=drq_robotiq_policy \
    --max_traj_length 100 \
    --camera_mode rgb \
    --seed 2 \
    --max_steps 20000 \
    --random_steps 100 \
    --training_starts 100 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 1000 \
    --encoder_type resnet-pretrained \
#    --debug