#!/bin/bash
#SBATCH --job-name=mlp-grid-search
#SBATCH -p short
#SBATCH --array=0-107%15
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=15:00:00
#SBATCH --output=logs/train_%A_%a.log

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

micromamba run -n torch_2026 python -u train_mlp.py 

echo "End:    $(date)"
