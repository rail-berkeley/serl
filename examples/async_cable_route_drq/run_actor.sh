export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaCableRoute-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_cable_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --encoder_type resnet-pretrained \
    --demo_path cable_route_20_demos_2024-01-04_12-10-54.pkl \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_cable_route_drq/10x10_30degs_20demos_rand_cable_096 \
    --reward_classifier_ckpt_path "/home/undergrad/code/serl_dev/examples/async_cable_route_drq/classifier_ckpt/" \
    --eval_checkpoint_step 20000 \
    --eval_n_trajs 20 \
    --max_traj_length 100 \
