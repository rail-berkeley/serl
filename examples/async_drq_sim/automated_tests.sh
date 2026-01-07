#!/bin/bash

SEEDS=$1
# WANDB_OUTPUT_DIR=~/wandb_logs
TEST="async_sac_state_sim.py"
CONDA_ENV="serl"
ENV="PandaReachSparseCube-v0"
MAX_STEPS=1000000
TRAINING_STARTS=1000
RANDOM_STEPS=1000
BATCH_SIZE=128
EXP_NAME="FIRST-TESTS-$ENV"
REPLAY_BUFFER_TYPE="fractal_symmetry_replay_buffer"
PRELOAD_RLDS="/data/data/serl/demos/franka_reach_drq_demo_script/10_demos_session_202500914_213515/PandaReachSparseCube-v0/0.1.0"
BASE_ARGS="--env $ENV --exp_name $EXP_NAME --training_starts $TRAINING_STARTS --random_steps $RANDOM_STEPS --batch_size $BATCH_SIZE --preload_rlds_path $PRELOAD_RLDS --encoder_type resnet-pretrained"
ARGS=""

function run_test {

    for seed in $(seq 1 1 $SEEDS)
    do
        # OPEN_PORTS=$( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 2 )
        # PORTS=( $OPEN_PORTS )
        # PORT_NUMBER=${PORTS[0]}
        # BROADCAST_PORT=${PORTS[1]}

        # ARGS+=" --port_number $PORT_NUMBER --broadcast_port $BROADCAST_PORT"

        echo "Running constant with args: $ARGS"
        tmux respawn-pane -k -t serl_session:0.1
        tmux respawn-pane -k -t serl_session:0.2
        tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash run_actor.sh --max_steps 2000000000 --seed $seed $BASE_ARGS $ARGS" C-m
        tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash run_learner.sh --max_steps $MAX_STEPS --seed $seed $BASE_ARGS $ARGS" C-m "exit" C-m

        # Wait for learner to finish
        while ! tmux capture-pane -t serl_session:0.2 -p | grep "logout" > /dev/null;
        do 
            sleep 100
        done
        echo "Finished!"
    done
}

# BASELINE TESTING
for replay_buffer_capacity in 1000000
do
    ARGS="--run_name baseline --replay_buffer_type memory_efficient_replay_buffer --replay_buffer_capacity $replay_buffer_capacity"
    run_test
done

# CONSTANT TESTING
for starting_branch_count in 1 27
do
    for workspace_width in 0.5
    do
        for replay_buffer_capacity in 1000000
        do
            ARGS="--run_name constant-$starting_branch_count^1 --replay_buffer_type $REPLAY_BUFFER_TYPE --replay_buffer_capacity $replay_buffer_capacity --workspace_width $workspace_width --branch_method 'constant' --starting_branch_count $starting_branch_count"
            run_test
        done
    done
done

tmux kill-window -t serl_session:$SEED
