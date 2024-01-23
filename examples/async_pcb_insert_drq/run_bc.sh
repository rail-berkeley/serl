export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../bc_policy.py "$@" \
    --env FrankaPCBInsert-Vision-v0 \
    --exp_name=serl_dev_bc_pcb_insert_random_resnet \
    --seed 0 \
    --batch_size 256 \
    --max_steps 20000 \
    --remove_xy True \
    --encoder_type resnet-pretrained \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/bc_demos/pcb_insert_20_demos_2024-01-20_15-33-14.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/bc_demos/pcb_insert_20_demos_2024-01-20_15-38-33.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/bc_demos/pcb_insert_20_demos_2024-01-20_15-44-18.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/bc_demos/pcb_insert_40_demos_2024-01-20_15-53-43.pkl \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/5x5_20degs_100demos_rand_pcb_insert_bc \
    --eval_checkpoint_step 20000 \
    --eval_n_trajs 100 \
    # --debug
