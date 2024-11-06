#!/usr/bin/env bash

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
OUTS="$BASE_DIR/results/scaling_experiments"
for NUM_NODES in 1 2 4 8 16 32; do
  export CONFIG_FILE="${BASE_DIR}/FourCastNet/config/AFNODIST$NUM_NODES.yaml"
  jobID=$(sbatch --job-name="Scal_$NUM_NODES" --nodes=$NUM_NODES --output="$OUTS/scale_$NUM_NODES.out" FourCastNet/scaling_experiments_worker.sh 2>&1 | sed 's/[S,a-z]* //g')
  sbatch --dependency=afterok:${jobID} --job-name="Fine_$NUM_NODES" --nodes=$NUM_NODES --output="$OUTS/scale_$NUM_NODES.finetune.out" FourCastNet/scaling_experiments_finetune.sh
done