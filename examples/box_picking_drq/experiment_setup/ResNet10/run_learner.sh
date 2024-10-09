export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name="ResNet10 one cam" \
    --camera_mode rgb \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/ResNet10/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/box_picking_20_demos_2024-08-20_rgb_depth.pkl \
    \
    --encoder_type resnet-pretrained \
    --state_mask all \
    --encoder_kwargs pooling_method \
    --encoder_kwargs spatial_softmax \
    --encoder_kwargs num_kp \
    --encoder_kwargs 128 \
#    --debug
