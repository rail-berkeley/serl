export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export ENV_NAME="FrankaPegInsert-Vision-v0" && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \

# Create checkpoint directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_drq_randomized.py "$@" \
    --learner \
    --env $ENV_NAME \
    --exp_name=serl-peg-insert \
    --seed 0 \
    --max_steps 25000 \
    --random_steps 1000 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 128 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_30_demos_2025-07-14_22-57-59.pkl\
    --checkpoint_period 1000 \
    --checkpoint_path "$CHECKPOINT_DIR" \
    #--debug # wandb is disabled when debug