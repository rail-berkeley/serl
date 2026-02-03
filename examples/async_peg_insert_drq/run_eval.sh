# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \
export CHECKPOINT_EVAL="/home/student/code/serl/examples/async_peg_insert_drq/checkpoints" && \


## Create checkpoint directory if it doesn't exist
#if [ ! -d "$CHECKPOINT_DIR" ]; then
#    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
#    mkdir -p "$CHECKPOINT_DIR" || {
#        echo "Failed to create checkpoint directory!" >&2
#        exit 1
#    }
#fi

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env "FrankaPegInsert-Vision-v0" \
    --eval_checkpoint_step 4500 \
    --eval_n_trajs 100 \
    --checkpoint_path "$CHECKPOINT_EVAL/baseline_03" \
