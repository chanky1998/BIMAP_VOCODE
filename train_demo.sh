#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Demo settings: keep these simple for classmates to read and change.
TRAIN_WAV_DIR="LibriSpeech_wav/demo/train"
VAL_WAV_DIR="LibriSpeech_wav/demo/dev"
CONFIG_FILE="cp_hifigan/v1_c512/config_demo_v1_512.json"
PRETRAINED_MODEL="cp_hifigan/v1_c512/g_07960000"
CHECKPOINT_DIR="cp_hifigan/v1_c512/checkpoints"

# train.py needs file lists without the .wav suffix.
for wav in "$TRAIN_WAV_DIR"/*.wav; do
    basename "${wav%.wav}"
done | sort > "LibriSpeech_wav/demo/training_demo.txt"

for wav in "$VAL_WAV_DIR"/*.wav; do
    basename "${wav%.wav}"
done | sort > "LibriSpeech_wav/demo/validation_demo.txt"

# Make a tiny config so the demo can finish quickly.
python - <<PY
import json

with open("config_v1_512.json", "r", encoding="utf-8") as f:
    config = json.load(f)

config["batch_size"] = 2
config["segment_size"] = 4096
config["num_workers"] = 0
config["num_gpus"] = 0

with open("cp_hifigan/v1_c512/config_demo_v1_512.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
    f.write("\n")
PY

echo "========== Train demo =========="
echo "Using demo data: $TRAIN_WAV_DIR"
echo "Saving checkpoints to: $CHECKPOINT_DIR"

python train.py \
    --config "$CONFIG_FILE" \
    --input_training_wavs_dir "$TRAIN_WAV_DIR" \
    --input_validation_wavs_dir "$VAL_WAV_DIR" \
    --input_training_file "LibriSpeech_wav/demo/training_demo.txt" \
    --input_validation_file "LibriSpeech_wav/demo/validation_demo.txt" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --history_checkpoint_path "cp_hifigan/v1_c512/checkpoint_history" \
    --tensorboard_logs_path "cp_hifigan/v1_c512/tensorboard" \
    --init_checkpoint "$PRETRAINED_MODEL" \
    --training_epochs 10 \
    --checkpoint_interval 10 \
    --summary_interval 5 \
    --validation_interval 10 \
    --stdout_interval 1

echo "Done. Latest checkpoint:"
ls -1 "$CHECKPOINT_DIR"/g_* | sort | tail -n 1
