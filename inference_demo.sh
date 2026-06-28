#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

TEST_WAV_DIR="LibriSpeech_wav/demo/test"
PRETRAINED_MODEL="cp_hifigan/v1_c512/g_07960000"
DEMO_MODEL="$(ls -1 cp_hifigan/v1_c512/checkpoints/g_* 2>/dev/null | sort | tail -n 1)"

if [ -z "$DEMO_MODEL" ]; then
    echo "No demo checkpoint found. Please run ./train_demo.sh first."
    exit 1
fi

echo "========== Inference demo =========="
echo "Using demo model: $DEMO_MODEL"

python inference.py \
    --input_wavs_dir "$TEST_WAV_DIR" \
    --output_dir generated_audios/demo_pretrained \
    --checkpoint_file "$PRETRAINED_MODEL" \
    --config_file config_v1_512.json \
    --experiment_name demo_pretrained \
    --csv_file demo_results.csv

python inference.py \
    --input_wavs_dir "$TEST_WAV_DIR" \
    --output_dir generated_audios/demo_finetuned \
    --checkpoint_file "$DEMO_MODEL" \
    --config_file cp_hifigan/v1_c512/config_demo_v1_512.json \
    --experiment_name demo_finetuned \
    --csv_file demo_results.csv

python inference.py \
    --input_wavs_dir "$TEST_WAV_DIR" \
    --output_dir generated_audios/demo_quantized \
    --checkpoint_file "$PRETRAINED_MODEL" \
    --config_file config_v1_512.json \
    --experiment_name demo_quantized \
    --csv_file demo_results.csv \
    --quantize \
    --quantize_scope resblocks_range \
    --calibration_samples 5 \
    --quantize_resblock_start 3 \
    --quantize_resblock_end 8

echo "Done. Results are in demo_results.csv"
echo "Generated wavs are in generated_audios/demo_pretrained and generated_audios/demo_finetuned"
