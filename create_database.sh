export OMP_NUM_THREADS=2

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=8 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1233 IDPerturb/sample.py