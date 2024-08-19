#!/bin/bash

#SBATCH --job-name=createAndCombine        # job name
#SBATCH --partition=single                 # queue for resource allocation
#SBATCH --time=2:00                        # wall-clock time limit
#SBATCH --mem=1000                        # memory
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --mail-type=NONE                   # Notify user by email when certain event types occur.


jobID=$(sbatch ./FourCastNet/combineNc.sh 2>&1 | sed 's/[S,a-z]* //g')
sbatch --dependency=afterok:${jobID} ./FourCastNet/Createh5.sh