export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/robotiq_drq/drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name="Depth Image Evaluation" \
    --camera_mode depth \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/experiment_setup/Depth Image/checkpoints Depth image 0821-15:14"\
    --eval_checkpoint_step 8000 \
    --eval_n_trajs 30 \
    \
    --encoder_type resnet-pretrained-18 \
    --state_mask all \
    --encoder_kwargs pooling_method \
    --encoder_kwargs spatial_softmax \
    --encoder_kwargs num_kp \
    --encoder_kwargs 128 \
#    --debug
