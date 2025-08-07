#!/usr/bin/env bash
#SBATCH -A PROJECT
#SBATCH -p CLUSTER
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-00:29:59

input_file=$1  # opting to also take the file as an input argument

# Read the given line from the input file and evaluate it:
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 matplotlib/3.7.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a
source /folder1/folder2/venv/bin/activate

echo "[`date`] Running partition=$partition"

python inference_lsun.py --index_partition=$partition