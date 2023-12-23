export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_rlpd_drq_sim.py \
    --learner \
    --env PandaPickCubeVision-v0 \
    --exp_name=serl_dev_rlpd_drq_sim_test_resnet \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 1000 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet \
    --demo_path franka_lift_cube_image_20_trajs.pkl \
    --debug
