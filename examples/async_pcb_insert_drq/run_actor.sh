export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaPCBInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --encoder_type resnet-pretrained \
    --demo_path pcb_insert_20_demos_2023-12-27_19-40-50.pkl \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/5x5_20degs_20demos_rand_pcb_insert_096 \
    --eval_checkpoint_step 20000 \
    --eval_n_trajs 100
