export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name=drq_robotiq_policy_evaluation \
    --max_traj_length 100 \
    --camera_mode depth \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/checkpoints 0523 rgb stable pooling=avg"\
    --eval_checkpoint_step 20000 \
    --encoder_type resnet-pretrained \
    --eval_n_trajs 10 \
    --debug