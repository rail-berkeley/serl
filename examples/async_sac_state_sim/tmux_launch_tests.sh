#!/bin/bash

SEEDS=5
# Create a new tmux session
tmux new-session -d -s serl_session
tmux setw -g remain-on-exit on

# Split the window horizontally
tmux split-window -v
tmux split-pane -h -t serl_session:0.1

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_session:0.0 "bash automated_tests.sh $SEEDS" C-m


# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
