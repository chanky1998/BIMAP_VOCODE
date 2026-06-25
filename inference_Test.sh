#!/bin/bash -l

#SBATCH --job-name=inference_512ch_h1-4
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

EXP_NAME="512 Channels (g_07960000)"
CONFIG_FILE="cp_hifigan/v1_c512/config.json"
CHECKPOINT_FILE="$HOME/hifi-gan/cp_hifigan/v1_c512/g_07960000"
CHECKPOINT_QUANTIZED="$HOME/hifi-gan/cp_hifigan/v1_c512"
QUANTIZED_SCOPE="resblocks"
QUANTIZE_RESBLOCK_START="3"
QUANTIZE_RESBLOCK_END="8"
CALIBRATION_SAMPLES="50"
CHECKPOINT_P30="$HOME/hifi-gan/cp_hifigan/v1_c512_ft30"
CHECKPOINT_P50="$HOME/hifi-gan/cp_hifigan/v1_c512_ft50"
CHECKPOINT_P70="$HOME/hifi-gan/cp_hifigan/v1_c512_ft70"
CHECKPOINT_P_FILE="g_00265000"
PRUNE_WAY="Mask_pruning"
CSV="experiments_results.csv"
CHANNEL="512C"


echo "==========(sbatch.tinyfat) Inference: ${EXP_NAME} =========="

echo "========== H2: quantized (INT8 dynamic quantization) =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H2_${CHANNEL}_int8 \
	--checkpoint_file ${CHECKPOINT_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H2_${CHANNEL}_quantized_${QUANTIZED_SCOPE} \
	--csv_file "${CSV}" \
    --quantize \
    --quantize_scope ${QUANTIZED_SCOPE} \
    --calibration_samples ${CALIBRATION_SAMPLES} \
    --quantize_resblock_start ${QUANTIZE_RESBLOCK_START} \
    --quantize_resblock_end ${QUANTIZE_RESBLOCK_END} \
    --save_compressed_checkpoint \
    --compressed_checkpoint_file ${CHECKPOINT_QUANTIZED}/compressed_checkpoint_int8

echo "========== H4: pruning 30% + quantize =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H4_${CHANNEL}_pruned30_int8 \
	--checkpoint_file ${CHECKPOINT_P30}/${CHECKPOINT_P_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H4_${CHANNEL}_pruned30_int8_${PRUNE_WAY}_${QUANTIZED_SCOPE} \
    --csv_file "${CSV}" \
    --quantize \
    --quantize_scope ${QUANTIZED_SCOPE} \
    --calibration_samples ${CALIBRATION_SAMPLES} \
    --quantize_resblock_start ${QUANTIZE_RESBLOCK_START} \
    --quantize_resblock_end ${QUANTIZE_RESBLOCK_END} \
    --save_compressed_checkpoint \
    --compressed_checkpoint_file ${CHECKPOINT_P30}/compressed_checkpoint_int8

echo "========== H4: pruning 50% + quantize =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H4_${CHANNEL}_pruned50_int8 \
	--checkpoint_file ${CHECKPOINT_P50}/${CHECKPOINT_P_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H4_${CHANNEL}_pruned50_int8_${PRUNE_WAY}_${QUANTIZED_SCOPE} \
    --csv_file "${CSV}" \
    --quantize \
    --quantize_scope ${QUANTIZED_SCOPE} \
    --calibration_samples ${CALIBRATION_SAMPLES} \
    --quantize_resblock_start ${QUANTIZE_RESBLOCK_START} \
    --quantize_resblock_end ${QUANTIZE_RESBLOCK_END} \
    --save_compressed_checkpoint \
    --compressed_checkpoint_file ${CHECKPOINT_P50}/compressed_checkpoint_int8

echo "========== H4: pruning 70% + quantize =========="
srun.tinyfat python inference.py \
    --output_dir generated_audios/generated_H4_${CHANNEL}_pruned70_int8 \
	--checkpoint_file ${CHECKPOINT_P70}/${CHECKPOINT_P_FILE} \
	--config_file ${CONFIG_FILE} \
    --experiment_name H4_${CHANNEL}_pruned70_int8_${PRUNE_WAY}_${QUANTIZED_SCOPE} \
    --csv_file "${CSV}" \
    --quantize \
    --quantize_scope ${QUANTIZED_SCOPE} \
    --calibration_samples ${CALIBRATION_SAMPLES} \
    --quantize_resblock_start ${QUANTIZE_RESBLOCK_START} \
    --quantize_resblock_end ${QUANTIZE_RESBLOCK_END} \
    --save_compressed_checkpoint \
    --compressed_checkpoint_file ${CHECKPOINT_P70}/compressed_checkpoint_int8

echo "Appended to ${CSV}:"
tail -n 1 "${CSV}" || true

echo "========== Done =========="
