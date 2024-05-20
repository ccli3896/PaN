#!/bin/bash
#
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G # Memory for ENTIRE JOB
#SBATCH --array=0-2499

#SBATCH -t 0-01:00 # time (D-HH:MM)

module load python/3.10.9-fasrc01
source activate virworm

python3 24_02_12_noisesweepJ10_in.py $SLURM_ARRAY_TASK_ID
