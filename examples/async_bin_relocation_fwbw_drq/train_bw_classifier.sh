export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python train_reward_classifier.py "$@" \
    --classifier_name bw \
    --positive_demo_paths ./classifier_data/bw_bin_relocate_338_front_cam_goal_2024-01-23_15-06-18.pkl \
    --positive_demo_paths ./classifier_data/bw_bin_relocate_400_front_cam_goal_2024-01-23_15-12-49.pkl \
    --negative_demo_paths ./classifier_data/bw_bin_relocate_475_front_cam_failed_2024-01-23_15-12-49.pkl \
    --negative_demo_paths ./classifier_data/bw_bin_relocate_762_front_cam_failed_2024-01-23_15-06-18.pkl \
    --negative_demo_paths ./classifier_data/bw_bin_relocate_2000_front_cam_failed_2024-01-23_15-26-35.pkl
