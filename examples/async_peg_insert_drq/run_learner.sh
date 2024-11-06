export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python3 async_drq_randomized.py "$@" \
    --learner \
    --env KukaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_097 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_20_demos_2023-12-25_16-13-25.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_peg_insert_drq/5x5_20degs_20demos_rand_peg_insert_097
