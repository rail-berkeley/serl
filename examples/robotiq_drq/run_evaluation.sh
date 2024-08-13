export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env_tests \
    --exp_name=drq_evaluation \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --encoder_type voxnet-pretrained \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/checkpoints 0809-15:53 pVoxNetOnly 1quat"\
    --eval_checkpoint_step 10000 \
    --eval_n_trajs 20 \
    --state_mask none \
    --debug
