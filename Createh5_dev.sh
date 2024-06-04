#!/bin/bash

#SBATCH --job-name=DEVcreateH5              # job name
#SBATCH --partition=dev_multiple                 # queue for resource allocation
#SBATCH --nodes=4                          # number of nodes to be used
#SBATCH --time=0:30:00                      # wall-clock time limit
#SBATCH --mem=90000                        # memory
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.

export PYDIR=./FourCastNet/data_process/
export VENVDIR=./.venvs/download

# Set up modules.
module purge                               # Unload all currently loaded modules.
module load compiler/intel/2021.4.0              # Load required modules.
module load devel/python/3.8.6_intel_19.1
module load devel/scorep/7.1-intel-2021.4.0-openmpi-4.1

source ${VENVDIR}/bin/activate # Activate your virtual environment.

mpirun --mca mpi_warn_on_fork 0 python -u ${PYDIR}/parallel_copy_RAM_MA.py   # Run your Python script.