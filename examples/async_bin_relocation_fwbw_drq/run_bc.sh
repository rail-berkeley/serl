export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../bc_policy.py "$@" \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_bc_bin_fwbw_resnet \
    --seed 0 \
    --batch_size 256 \
    --max_steps 30000 \
    --gripper \
    --encoder_type resnet-pretrained \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bc_demos/bc_bin_relocate_20_demos_2024-01-24_17-17-55.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bc_demos/bc_bin_relocate_20_demos_2024-01-24_17-41-02.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bc_demos/bc_bin_relocate_20_demos_2024-01-24_17-44-58.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bc_demos/bc_bin_relocate_20_demos_2024-01-24_17-51-50.pkl \
    --demo_paths /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bc_demos/bc_bin_relocate_20_demos_2024-01-24_17-56-19.pkl \
    --checkpoint_path /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bin_fwbw_bc \
    --eval_checkpoint_step 30000 \
    --eval_n_trajs 100 \
    # --debug
