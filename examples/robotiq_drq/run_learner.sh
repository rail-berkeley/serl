export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python drq_policy_robotiq.py "$@" \
    --learner \
    --env robotiq_camera_env \
    --exp_name=drq_robotiq_policy \
    --camera_mode rgb \
    --max_traj_length 100 \
    --seed 0 \
    --max_steps 20000 \
    --random_steps 0 \
    --training_starts 0 \
    --utd_ratio 4 \
    --batch_size 128 \
    --eval_period 1000 \
    --encoder_type resnet-pretrained \
    --checkpoint_period 10000 \
    --demo_path /home/nico/real-world-rl/serl/examples/robotiq_drq/pcb_insert_20_demos_apr30_newrew.pkl \
    --debug
