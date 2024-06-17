export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --debug
