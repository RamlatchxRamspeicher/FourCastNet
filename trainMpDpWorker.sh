#!/usr/bin/env bash 

#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

#SBATCH -p dev_accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive
#SBATCH --mail-type=ALL

ml purge
ml restore MA41

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
config_file="${BASE_DIR}/FourCastNet/config/AFNOMUDP_Devel.yaml"
config=$1 
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
srun -u --mpi=pmix bash -c " 
    source $DDP_VARS
    python $TRAIN_FILE --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
