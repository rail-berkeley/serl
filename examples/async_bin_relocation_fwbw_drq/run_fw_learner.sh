export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096_fw \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw fw \
    --demo_path ./demos/fw_bin_2000_demo_2024-01-23_18-49-56.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bin_fw_096
