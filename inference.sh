#!/bin/bash -l

#SBATCH --job-name=inference_128ch_h1-4
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

echo "========== H1: baseline (no compression) =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H1_${CHANNEL} \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H1_${CHANNEL}_baseline \
	--csv_file "${CSV}"

echo "========== H2: quantized (INT8 dynamic quantization) =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H2_${CHANNEL}_int8 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H2_${CHANNEL}_quantized \
	--csv_file "${CSV}" \
    --quantize

echo "========== H3: pruning 30% =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H3_${CHANNEL}_pruned30 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H3_${CHANNEL}_pruned30 \
	--csv_file "${CSV}" \
    --prune_ratio 0.3 

echo "========== H3: pruning 50% =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H3_${CHANNEL}_pruned50 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H3_${CHANNEL}_pruned50 \
	--csv_file "${CSV}" \
    --prune_ratio 0.5 

echo "========== H3: pruning 70% =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H3_${CHANNEL}_pruned70 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H3_${CHANNEL}_pruned70 \
	--csv_file "${CSV}" \
    --prune_ratio 0.7 

echo "========== H4: pruning 30% + quantize =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H4_${CHANNEL}_pruned30_int8 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H4_${CHANNEL}_pruned30_int8 \
    --csv_file "${CSV}" \
    --prune_ratio 0.3 \
    --quantize 

echo "========== H4: pruning 50% + quantize =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H4_${CHANNEL}_pruned50_int8 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H4_${CHANNEL}_pruned50_int8 \
    --csv_file "${CSV}" \
    --prune_ratio 0.5 \
    --quantize 

echo "========== H4: pruning 70% + quantize =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H4_${CHANNEL}_pruned70_int8 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H4_${CHANNEL}_pruned70_int8 \
    --csv_file "${CSV}" \
    --prune_ratio 0.7 \
    --quantize 

echo "Appended to ${CSV}:"
tail -n 1 "${CSV}" || true

echo "========== Done =========="
