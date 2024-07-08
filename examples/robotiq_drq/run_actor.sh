export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --max_traj_length 100 \
    --exp_name=drq_10box \
    --camera_mode pointcloud \
    --seed 1 \
    --max_steps 30000 \
    --random_steps 500 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 2500 \
    --encoder_type voxnet \
#    --debug
