export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../bc_policy.py "$@" \
    --env FrankaCableRoute-Vision-v0 \
    --exp_name=serl_dev_bc_cable_random_resnet \
    --seed 0 \
    --batch_size 256 \
    --max_steps 20000 \
    --remove_xy True \
    --encoder_type resnet-pretrained \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_cable_route_drq/bc_demos/cable_route_10_demos_2024-01-20_14-39-46.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_cable_route_drq/bc_demos/cable_route_20_demos_2024-01-20_14-44-26.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_cable_route_drq/bc_demos/cable_route_70_demos_2024-01-20_14-58-22.pkl \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_cable_route_drq/10x10_30degs_100demos_rand_cable_route_bc \
    --eval_checkpoint_step 20000 \
    --eval_n_trajs 100 \
    --reward_classifier_ckpt_path "/home/undergrad/code/serl_dev/examples/async_cable_route_drq/classifier_ckpt" \
    # --debug
