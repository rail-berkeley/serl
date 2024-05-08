export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 1000 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --debug
