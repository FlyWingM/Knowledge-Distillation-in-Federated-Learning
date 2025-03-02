#!/bin/bash

#SBATCH --account=Berzelius-2024-101        # Specify the SLURM account
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --time=3-00:00:00                   # Job duration, e.g., 1 day
##SBATCH --reservation=safe
#SBATCH --job-name=fl_base                     # Job name
#SBATCH --output=/proj/seo-220318/myfl/2408/source/sh/fl_base_%j.out  # Standard output file
#SBATCH --error=/proj/seo-220318/myfl/2408/source/sh/fl_base_%j.err   # Standard error file

# Load necessary modules and activate environments
module load Anaconda/2021.05-nsc1
conda activate fl_a100

# Use environment variables for the model name and pre_distr value
MODEL_NAME=${MODEL_NAME:-resnet34}
PRE_DISTR=${PRE_DISTR:-d06}

# Your command(s) for job1, using the model name and pre_distr from the environment variables
python3 -u /proj/seo-220318/myfl/2408/source/server.py --data cinic10 --nn ${MODEL_NAME} --pre_distr ${PRE_DISTR}
