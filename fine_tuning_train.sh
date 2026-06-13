#!/bin/bash -l

#SBATCH --job-name=fine_tuning_512ch_30%
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --output=fine_tuning_logs/%x-%j.out
#SBATCH --error=fine_tuning_logs/%x-%j.err
#SBATCH --mail-user=jinying.chen@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV


source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate hifi-gan

cd $HOME/hifi-gan

echo "========== Fine-Tuning Model 512 Channels =========="
srun python train.py --config config_v1_512.json \
    --checkpoint_path $HOME/hifi-gan/cp_hifigan/v1_c512_ft30 \
    --history_checkpoint_path $WORK/cp_hifigan/v1_c512_ft30 \
    --tensorboard_logs_path $WORK/logs/v1_c512_ft30 \
    --prune_ratio 0.3 \
    --training_epochs 50 \
    --checkpoint_interval 10 \
    --summary_interval 500 \
    --validation_interval 100 \
#    --init_checkpoint $HOME/hifi-gan/cp_hifigan/v1_c512/g_07640000