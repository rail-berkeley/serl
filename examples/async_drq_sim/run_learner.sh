export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_sim.py "$@" \
    --learner \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 1000 \
    --utd_ratio 4 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    # --demo_path franka_lift_cube_image_20_trajs.pkl \
    --debug # wandb is disabled when debug
