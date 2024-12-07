export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name=ox_picking \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 20000 \
    --checkpoint_period 1000 \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --demo_path "demo path here *.pkl" \
#    --enable_obs_rotation_wrapper \
#    --debug
