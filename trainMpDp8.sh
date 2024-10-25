#!/usr/bin/env bash 

#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4

#SBATCH -J 8MP
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/241025/fcn_mpdp_8_wandb.out"
#SBATCH -p accelerated
#SBATCH --mem=501600mb
#SBATCH --mail-type=ALL

ml purge
ml restore MA41

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
config_file="${BASE_DIR}/FourCastNet/config/AFNOMUDP8.yaml"
config="afno_backbone" 
run_num="12"

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
# export WANDB_MODE=offline
export WANDB_START_METHOD="thread"
export WANDB_API_KEY=239f4e150ee486bc13a02cacd1c44c40d8556b73
# export OMPI_MCA_btl_tcp_max_send_size=400131200

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
