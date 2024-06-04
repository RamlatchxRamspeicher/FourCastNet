#!/usr/bin/env bash 

#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1

#SBATCH -J FCN_Test
#SBATCH --output="/hkfs/work/workspace/scratch/vm6493-fourcastnet/fourcastnet/results/slurm_logs/test/slurm-%j"
#SBATCH -p accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive

ml purge
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

BASE_DIR="/hkfs/work/workspace/scratch/"
config_file="${BASE_DIR}/fourcastnet/config/AFNO_test.yaml"
config="afno_backbone_finetune" 
run_num="0"

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

SRUN_PARAMS=(
  --mpi="pmi2"
  --gpus-per-task=1
  --gpu-bind="closest"
  --label
)


TRAIN_FILE="$BASE_DIR/fourcastnet/train_test.py"
DDP_VARS="$BASE_DIR/fourcastnet/export_DDP_vars.sh"

source $BASE_DIR/venv/bin/activate
srun -u --mpi=pmi2 bash -c " 
  source $DDP_VARS
  python $TRAIN_FILE --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"