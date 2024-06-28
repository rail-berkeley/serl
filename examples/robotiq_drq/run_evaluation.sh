export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env_tests \
    --exp_name=drq_evaluation \
    --max_traj_length 100 \
    --camera_mode rgb \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/checkpoints 0628-14:59 10box res18 avg"\
    --eval_checkpoint_step 20000 \
    --encoder_type resnet-pretrained-18 \
    --eval_n_trajs 20 \
    --debug
