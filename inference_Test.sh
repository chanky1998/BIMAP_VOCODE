#!/bin/bash -l

#SBATCH --job-name=inference_128ch_test
#SBATCH --clusters=tinyfat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=work
#SBATCH --hint=nomultithread
#SBATCH --output=inference_logs/%x-%j.out
#SBATCH --error=inference_logs/%x-%j.err
#SBATCH --mail-user=jinying.chen@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate hifi-gan

cd $HOME/hifi-gan
export CUDA_VISIBLE_DEVICES=""

EXP_NAME="128 Channels (g_05540000)"
CONFIG_FILE="cp_hifigan/v1_c128/config.json"
CHECKPOINT_FILE="$HOME/hifi-gan/cp_hifigan/v1_c128/g_05540000"
CSV="experiments_results.csv"
CHANNEL="128C"


echo "==========(sbatch.tinyfat) Inference: ${EXP_NAME} =========="


echo "========== H3: pruning 0% (test)=========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H3_${CHANNEL}_pruned10 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H3_${CHANNEL}_pruned10 \
	--csv_file "${CSV}" \
    --prune_ratio 0.1 

echo "========== Done =========="
