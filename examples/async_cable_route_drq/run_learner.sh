export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaCableRoute-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_cable_random_resnet_096 \
    --seed 0 \
    --random_steps 600 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path cable_route_20_demos_2024-01-04_12-10-54.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_cable_route_drq/10x10_30degs_20demos_rand_cable_096
