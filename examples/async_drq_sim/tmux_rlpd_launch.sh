#!/bin/bash

# use the default values if the env variables are not set
EXAMPLE_DIR=${EXAMPLE_DIR:-"examples/async_drq_sim"}
CONDA_ENV=${CONDA_ENV:-"serl"}

cd $EXAMPLE_DIR
echo "Running from $(pwd)"

# check if the pkl file exists, else download it
RESNET_FILE="resnet10_params.pkl"
if [ ! -f "$RESNET_FILE" ]; then
    wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
fi

# download trajectory data for offline RL
DATA_FILE="franka_lift_cube_image_20_trajs.pkl"
if [ ! -f "$DATA_FILE" ]; then
    wget https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl
fi

# check if both file exists else throw error
if [ ! -f "$RESNET_FILE" ] || [ ! -f "$DATA_FILE" ]; then
    echo "Error: $RESNET_FILE or $DATA_FILE does not exist"
    exit 1
fi

# Create a new tmux session
tmux new-session -d -s serl_session

# Split the window vertically
tmux split-window -v

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_session:0.0 "conda activate $CONDA_ENV && bash run_actor.sh" C-m

# Navigate to the activate the conda environment in the second pane
tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash run_learner.sh --demo_path franka_lift_cube_image_20_trajs.pkl" C-m

# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
