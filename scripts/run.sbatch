#!/bin/bash

#SBATCH --job-name=kaputt
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err

echo "Used nodes: $SLURM_NODELIST"


unset UENV_MOUNT_LIST
export OMP_NUM_THREADS=1
export HF_HOME=$SCRATCH/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO 
MASTER_ADDR=$(hostname)
MASTER_PORT=25678
GPUS_PER_NODE=4
LAUNCHER="torchrun \
	--node-rank=\${SLURM_PROCID} \
	--master-addr=$MASTER_ADDR \
	--master-port=$MASTER_PORT \
	--nnodes=$SLURM_NNODES \
	--nproc-per-node=$GPUS_PER_NODE"
	

srun -ul --environment=llm bash -c "
	cd /users/ahernnde/workspace/repos/kaputt
	$LAUNCHER scripts/stress.py
	"
