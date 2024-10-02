#!/usr/bin/env bash 

#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

#SBATCH -J DevOrig
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/.spike/results/300924/fcn_mpdp_test_2.out"
#SBATCH -p dev_accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive
#SBATCH --mail-type=NONE

ml purge
ml restore MA41

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
config_file="${BASE_DIR}/FourCastNet/config/AFNOMUDP_Devel.yaml"
config="afno_backbone_two" 
run_num="0"

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export WANDB_MODE=offline
export WANDB_START_METHOD="thread"

# SRUN_PARAMS=(
#   --mpi="pmi2"
#   --gpus-per-task=1
#   --gpu-bind="closest"
#   --label
# )


TRAIN_FILE="$BASE_DIR/FourCastNet/train.py"
DDP_VARS="$BASE_DIR/FourCastNet/export_DDP_vars.sh"

source $BASE_DIR/.venvs/Fourcastv2/bin/activate
# jobID=$(
srun -u --mpi=pmix bash -c " 
    source $DDP_VARS
    python $TRAIN_FILE --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num" # 2>&1 | sed 's/[S,a-z]* //g'

# )

#for i in {'two','four','eight'}; do
#  config="afno_backbone_$i"
#  sbatch --dependency=afterok:${jobID} --job-name="Dev_$i" --output="/hkfs/work/workspace/scratch/ie5012-MA/.spike/results/250924/fcn_mpdp_test_$i.out" \
#  FourCastNet/trainMpDpWorker.sh $config
#done