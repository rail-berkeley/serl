#!/bin/bash

EXAMPLE_DIR="examples/async_drq_sim"
CONDA_ENV="serl"

cd $EXAMPLE_DIR
echo "Running from $(pwd)"

# check if the pkl file exists, else download it
# NOTE: this will not work when repo is private
FILE="resnet10_params.pkl"
if [ ! -f "$FILE" ]; then
    wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
fi

# Create a new tmux session
tmux new-session -d -s serl_session

# Split the window vertically
tmux split-window -v

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_session:0.0 "conda activate $CONDA_ENV && bash run_actor.sh" C-m

# Navigate to the activate the conda environment in the second pane
tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash run_learner.sh" C-m

# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
