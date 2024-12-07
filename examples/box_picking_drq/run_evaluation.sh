export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env \
    --exp_name=drq_evaluation \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "checkpoint folder path here"\
    --eval_checkpoint_step 10000 \
    --eval_n_trajs 20 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --enable_obs_rotation_wrapper \
#    --debug
