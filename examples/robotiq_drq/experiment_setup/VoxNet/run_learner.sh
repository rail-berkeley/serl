export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/nico/real-world-rl/serl/examples/robotiq_drq/drq_policy_robotiq.py "$@" \
    --learner \
    --env robotiq_camera_env_tests \
    --exp_name="voxnet only" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 200 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 2500 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/robotiq_drq/experiment_setup/VoxNet/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/robotiq_drq/pcb_insert_20_demos_jul29_noforcetorque.pkl \
    \
    --encoder_type voxnet \
    --state_mask none \
    --encoder_bottleneck_dim 128 \
#    --debug
