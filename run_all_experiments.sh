#!/bin/bash
# Batch script to run H1-H4 experiments with CSV logging
# Usage: bash run_all_experiments.sh

set -e

# Configuration
REFERENCE_DIR="LibriSpeech_wav/test"
CSV_FILE="experiments_results.csv"
CHECKPOINT_STEP="g_04735000"  # Adjust if you want to use a different checkpoint

echo "=========================================="
echo "HiFi-GAN Compression Experiments (H1-H4)"
echo "=========================================="

# H1: Model Scaling (baseline models, no compression)
echo -e "\n>>> H1: Model Scaling Effect"

for CONFIG in config_v1_128.json config_v1_256.json config_v1_512.json; do
    CHANNEL=$(echo $CONFIG | grep -oE '[0-9]+' | head -1)
    CHECKPOINT="cp_hifigan/v1_c${CHANNEL}/${CHECKPOINT_STEP}"
    EXP_NAME="H1_${CHANNEL}C_baseline"
    
    if [ -f "$CHECKPOINT" ]; then
        echo "Running: $EXP_NAME"
        python inference.py \
            --input_wavs_dir "$REFERENCE_DIR" \
            --output_dir "generated_audios/generated_H1_${CHANNEL}C" \
            --checkpoint_file "$CHECKPOINT" \
            --config_file "$CONFIG" \
            --experiment_name "$EXP_NAME" \
            --csv_file "$CSV_FILE"
    fi
done

# H2: Quantization Effect
echo -e "\n>>> H2: Quantization Effect"

for CONFIG in config_v1_128.json config_v1_256.json config_v1_512.json; do
    CHANNEL=$(echo $CONFIG | grep -oE '[0-9]+' | head -1)
    CHECKPOINT="cp_hifigan/v1_c${CHANNEL}/${CHECKPOINT_STEP}"
    EXP_NAME="H2_${CHANNEL}C_quantized"
    
    if [ -f "$CHECKPOINT" ]; then
        echo "Running: $EXP_NAME"
        python inference.py \
            --input_wavs_dir "$REFERENCE_DIR" \
            --output_dir "generated_audios/generated_H2_${CHANNEL}C_int8" \
            --checkpoint_file "$CHECKPOINT" \
            --config_file "$CONFIG" \
            --experiment_name "$EXP_NAME" \
            --csv_file "$CSV_FILE" \
            --quantize
    fi
done

# H3: Pruning Effect
echo -e "\n>>> H3: Pruning Effect (30%, 50%, 70%)"

for PRUNE_RATIO in 0.3 0.5 0.7; do
    for CONFIG in config_v1_128.json config_v1_256.json config_v1_512.json; do
        CHANNEL=$(echo $CONFIG | grep -oE '[0-9]+' | head -1)
        CHECKPOINT="cp_hifigan/v1_c${CHANNEL}/${CHECKPOINT_STEP}"
        PRUNE_PCT=$((${PRUNE_RATIO%.*}0))
        EXP_NAME="H3_${CHANNEL}C_pruned${PRUNE_PCT}"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Running: $EXP_NAME"
            python inference.py \
                --input_wavs_dir "$REFERENCE_DIR" \
                --output_dir "generated_audios/generated_H3_${CHANNEL}C_pruned${PRUNE_PCT}" \
                --checkpoint_file "$CHECKPOINT" \
                --config_file "$CONFIG" \
                --experiment_name "$EXP_NAME" \
                --csv_file "$CSV_FILE" \
                --prune_ratio "$PRUNE_RATIO" \
                --save_compressed_checkpoint \
                --compressed_checkpoint_file "cp_hifigan/v1_c${CHANNEL}/v1_c${CHANNEL}_pruned${PRUNE_PCT}"
        fi
    done
done

# H4: Combined Compression (Pruning + Quantization)
echo -e "\n>>> H4: Combined Compression (Pruning + Quantization)"

for PRUNE_RATIO in 0.3 0.5 0.7; do
    for CONFIG in config_v1_128.json config_v1_256.json config_v1_512.json; do
        CHANNEL=$(echo $CONFIG | grep -oE '[0-9]+' | head -1)
        CHECKPOINT="cp_hifigan/v1_c${CHANNEL}/${CHECKPOINT_STEP}"
        PRUNE_PCT=$((${PRUNE_RATIO%.*}0))
        EXP_NAME="H4_${CHANNEL}C_pruned${PRUNE_PCT}_int8"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Running: $EXP_NAME"
            python inference.py \
                --input_wavs_dir "$REFERENCE_DIR" \
                --output_dir "generated_audios/generated_H4_${CHANNEL}C_pruned${PRUNE_PCT}_int8" \
                --checkpoint_file "$CHECKPOINT" \
                --config_file "$CONFIG" \
                --experiment_name "$EXP_NAME" \
                --csv_file "$CSV_FILE" \
                --prune_ratio "$PRUNE_RATIO" \
                --quantize \
                --save_compressed_checkpoint \
                --compressed_checkpoint_file "cp_hifigan/v1_c${CHANNEL}/v1_c${CHANNEL}_pruned${PRUNE_PCT}_int8"
        fi
    done
done

echo -e "\n=========================================="
echo "All experiments completed!"
echo "Results saved to: $CSV_FILE"
echo "=========================================="

# Display CSV summary
echo -e "\nResults Summary:"
echo "========================================"
if [ -f "$CSV_FILE" ]; then
    column -t -s, "$CSV_FILE" | head -20
fi
