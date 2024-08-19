#!/bin/bash

#SBATCH --job-name=createH5              # job name
#SBATCH --partition=multiple                 # queue for resource allocation
#SBATCH --time=20:00:00                      # wall-clock time limit
#SBATCH --mem=90000                        # memory
#SBATCH --nodes=20                          # number of nodes to be used
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.

export PYDIR=./FourCastNet/data_process/
export VENVDIR=./.venvs/download

# Set up modules.
module purge                               # Unload all currently loaded modules.
module load compiler/gnu/10.2              # Load required modules.
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2
module load lib/hdf5/1.12.2-gnu-10.2-openmpi-4.1

source ${VENVDIR}/bin/activate # Activate your virtual environment.

python -u ${PYDIR}/parallel_copy_RAM_MA.py   # Run your Python script.