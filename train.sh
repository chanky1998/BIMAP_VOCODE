#!/bin/bash -l

#SBATCH --job-name=train_512ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --output=train_logs/%x-%j.out
#SBATCH --error=train_logs/%x-%j.err
#SBATCH --mail-user=jinying.chen@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV


source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate hifi-gan

cd $HOME/hifi-gan

echo "========== Training Model 512 Channels =========="
srun python train.py --config config_v1_512.json --checkpoint_path cp_hifigan/v1_test01 --history_checkpoint_path $WORK/cp_hifigan/v1_test01