#!/bin/bash 

#SBATCH -t 1:00:00                      # wall-clock time limit
#SBATCH --gres=gpu:4
#SBATCH -N 1                           # number of nodes to be used
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks-per-node=1

#SBATCH -J Testrun
#SBATCH --output results/logs/slurm_afno_backbone_finetune-%j.out
#SBATCH -p dev_accelerated               # queue for resource allocation
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.

# Set up modules.
module purge                               # Unload all currently loaded modules.
module load compiler/intel/2023.1.0              # Load required modules.
module load mpi/openmpi/4.1


BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
VENVDIR=$BASE_DIR/.venvs/Fourcastv2

config_file=$BASE_DIR/FourCastNet/config/AFNO.yaml
config='afno_backbone'
run_num='0'


export WANDB_MODE=offline
export WANDB_START_METHOD="thread"
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

SRUN_PARAMS=(
  --mpi="pmi2"
  --gpus-per-task=1
  --gpu-bind="closest"
  --label
)

TRAIN_FILE="$BASE_DIR/FourCastNet/train.py"
DDP_VARS="$BASE_DIR/FourCastNet/export_DDP_vars.sh"

source $VENVDIR/bin/activate # Activate your virtual environment.

srun -u --mpi=pmi2 bash -c " 
  source $DDP_VARS
  python $TRAIN_FILE --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
