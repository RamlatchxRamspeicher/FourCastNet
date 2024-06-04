#!/bin/bash

#SBATCH --job-name=DEVcreateH5              # job name
#SBATCH --partition=single                 # queue for resource allocation
#SBATCH --time=5:00:00                      # wall-clock time limit
#SBATCH --mem=180000                        # memory
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.

export PYDIR=./FourCastNet/data_process/
export VENVDIR=./.venvs/download

# Set up modules.
module purge                               # Unload all currently loaded modules.

source ${VENVDIR}/bin/activate # Activate your virtual environment.

python -u ${PYDIR}/copy_RAM_MA.py   # Run your Python script.