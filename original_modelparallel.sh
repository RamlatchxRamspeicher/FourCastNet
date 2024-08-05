#!/usr/bin/env bash 

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4

#SBATCH -J FCN_MP
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/010824/fcn_mp_net.out"
#SBATCH -p accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive

ml purge
ml restore MA41

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
config_file="${BASE_DIR}/FourCastNet/config/AFNO.yaml"
config="afno_model_parallel" 
run_num="0"

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

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