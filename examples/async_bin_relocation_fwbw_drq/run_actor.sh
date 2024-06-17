export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096 \
    --seed 0 \
    --random_steps 200 \
    --encoder_type resnet-pretrained \
    --demo_path fw_bin_2000_demo_2024-01-23_18-49-56.pkl \
    --fw_ckpt_path /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bin_fw_096 \
    --bw_ckpt_path /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bin_bw_096 \
    --fw_reward_classifier_ckpt_path "/home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/fw_classifier_ckpt" \
    --bw_reward_classifier_ckpt_path "/home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bw_classifier_ckpt" \
    --eval_checkpoint_step 31000 \
    --eval_checkpoint_step 100
