export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name=d \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 2 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 20000 \
    --encoder_type voxnet-pretrained \
    --checkpoint_period 2500 \
    --demo_path /home/nico/real-world-rl/serl/examples/box_picking_drq/box_picking_20_demos_aug9_noFT_1quad.pkl \
#    --debug
