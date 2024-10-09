export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_eval \
    --exp_name="ResNet10 Evaluation" \
    --camera_mode rgb \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_picking_drq/experiment_setup/ResNet10/checkpoints ResNet10 one cam 0821-11:18"\
    --eval_checkpoint_step 15000 \
    --eval_n_trajs 30 \
    \
    --encoder_type resnet-pretrained \
    --state_mask all \
    --encoder_kwargs pooling_method \
    --encoder_kwargs spatial_softmax \
    --encoder_kwargs num_kp \
    --encoder_kwargs 128 \
#    --debug
