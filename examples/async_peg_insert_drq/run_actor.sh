export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform &&\
python3 async_drq_randomized.py "$@" \
    --actor 1\
    --render \
    --env KukaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 200 \
    --training_starts 300 \
    --encoder_type resnet-pretrained \
    --demo_path /home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/rect_peg_skips_sdf_action6_surface_relaxed.pkl \
    --eval_checkpoint_step 16500 \
    --loaded_checkpoint_step 16500 \
    --eval_n_trajs 11 \
    --load_checkpoint_path /home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/checkpoints/ \
    --checkpoint_path /home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/checkpoints_new/ \
    --load_checkpoint 1 \
