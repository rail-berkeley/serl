export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform &&\
python3 async_drq_randomized.py "$@" \
    --learner 1\
    --env KukaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_097 \
    --seed 0 \
    --random_steps 200 \
    --training_starts 300 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path /home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/rect_peg_skips_sdf_action6_surface_relaxed.pkl \
    --checkpoint_period 300 \
    --loaded_checkpoint_step 1200 \
    --load_checkpoint_path /home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/checkpoints/ \
    --checkpoint_path /home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/checkpoints_new/ \
    --load_checkpoint 1 \
    
