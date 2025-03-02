#!/bin/bash

#SBATCH --account=Berzelius-2024-361  # Specify the SLURM account
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=3-00:00:00             # Job duration
#SBATCH --job-name=job_${MODEL_NAME}_${PRE_DISTR}  # Job name, includes model and distribution
#SBATCH --output=logs/%x-%j.out       # Standard output
#SBATCH --error=logs/%x-%j.err        # Standard error

# Load necessary modules and activate the Conda environment
module load Mambaforge/23.3.1-1-hpc1-bdist || { echo "Module load failed"; exit 1; }
conda activate fl_a100 || { echo "Conda environment activation failed"; exit 1; }

# Use environment variables for flexibility
CRITERION=${CRITERION:-acc}
PRE_DISTR=${PRE_DISTR:-d06}
ITER=${ITER:-1}
MODEL_NAME=${MODEL_NAME:-densenet121}

# Validate critical variables
if [[ -z "$CRITERION" || -z "$PRE_DISTR" || -z "$ITER" || -z "$MODEL_NAME" ]]; then
    echo "Error: Missing required environment variables."
    exit 1
fi

# Execute the Python script
python3 /proj/seo-220318/myfl/kfl/source/learning.py \
    --c_sel_cri "${CRITERION}" \
    --pre_distr "${PRE_DISTR}" \
    --l_iter_group "${ITER}" \
    --nn "${MODEL_NAME}" || { echo "Python script execution failed"; exit 1; }
