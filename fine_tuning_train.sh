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

CHANNEL="512"
PRUNNING="30"

echo "========== Fine-Tuning Model ${CHANNEL} Channels =========="
srun python train.py --config config_v1_${CHANNEL}_p${PRUNNING}.json \
    --checkpoint_path $HOME/hifi-gan/cp_hifigan/v1_c${CHANNEL}_physical${PRUNNING} \
    --history_checkpoint_path $WORK/cp_hifigan/v1_c${CHANNEL}_physical${PRUNNING} \
    --tensorboard_logs_path $WORK/logs/v1_c${CHANNEL}_physical${PRUNNING} \
    --training_epochs 150 \
    --checkpoint_interval 5000 \
    --summary_interval 500 \
    --validation_interval 1000 \
#    --init_checkpoint $HOME/hifi-gan/cp_hifigan/v1_c${CHANNEL}_ft${PRUNNING}/g_00265000