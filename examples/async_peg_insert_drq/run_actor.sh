export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_20_demos_2023-12-25_16-13-25.pkl \
    # --checkpoint_path /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/5x5_20degs_100demos_rand_pcb_insert_bc \
    # --eval_checkpoint_step 20000 \
    # --eval_n_trajs 100 \
