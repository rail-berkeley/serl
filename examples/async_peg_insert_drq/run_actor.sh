export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export ENV_NAME="FrankaPegInsert-Vision-v0" && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \
export CHECKPOINT_EVAL="/home/student/code/cleiver/serl/examples/async_peg_insert_drq/checkpoints/checkpoints-07-14-2025-23-15-59" && \


# Create checkpoint directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env $ENV_NAME \
    --exp_name=serl-peg-insert \
    --max_steps 25000 \
    --seed 0 \
    --random_steps 0 \
    --training_starts 200 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_30_demos_2025-07-14_22-57-59.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path "$CHECKPOINT_DIR" \
    # --eval_checkpoint_step=5000 \
    # --eval_n_trajs=5 \
    #--debug # wandb is disabled when debug
