export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export MUJOCO_GL=egl && \
export TF_GPU_ALLOCATOR=cuda_malloc_async && \

python async_drq_sim.py --actor "$@"
