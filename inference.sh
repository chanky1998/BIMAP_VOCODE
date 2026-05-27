#!/bin/bash -l

#SBATCH --job-name=inference_512ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=v100
#SBATCH --output=inference_logs/%x-%j.out
#SBATCH --error=inference_logs/%x-%j.err
#SBATCH --mail-user=jinying.chen@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate hifi-gan

cd $HOME/hifi-gan
export CUDA_VISIBLE_DEVICES=""

EXP_NAME="H1_512C_baseline"
OUTPUT_DIR="generated_audios/generated_files_LibriSpeech_wav"
CSV="experiments_results.csv"


echo "========== Inference: ${EXP_NAME} =========="


# Run inference and write directly into the global CSV
srun python inference.py \
	--config_file cp_hifigan/v1_test01/config.json \
	--checkpoint_file $HOME/hifi-gan/cp_hifigan/v1_test01/g_04735000 \
	--output_dir ${OUTPUT_DIR} \
	--experiment_name "${EXP_NAME}" \
	--csv_file "${CSV}"

echo "Appended to ${CSV}:"
tail -n 1 "${CSV}" || true

echo "========== Done =========="
