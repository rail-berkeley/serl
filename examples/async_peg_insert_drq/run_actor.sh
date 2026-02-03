# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \
export CHECKPOINT_EVAL="/home/student/code/serl/examples/async_peg_insert_drq/checkpoints/checkpoints-07-14-2025-23-15-59" && \


# # Create checkpoint directory if it doesn't exist
# if [ ! -d "$CHECKPOINT_DIR" ]; then
#     echo "Creating checkpoint directory: $CHECKPOINT_DIR"
#     mkdir -p "$CHECKPOINT_DIR" || {
#         echo "Failed to create checkpoint directory!" >&2
#         exit 1
#     }
# fi

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env "FrankaPegInsert-Vision-v0" \
    --random_steps 0 \
    --seed 4 \
    --training_starts 200 \
    --save_model \
    --max_steps 15000 \
    --encoder_type resnet-pretrained \
