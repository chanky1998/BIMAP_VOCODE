from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
from utils import (
    count_parameters,
    file_size_in_mb,
    prune_conv_layers,
    quantize_dynamic_model,
    save_checkpoint,
    collect_pairs,
    load_audio,
    align_waveforms,
    compute_mel,
    resample_audio,
    pesq_fn,
    stoi_fn,
)
import csv
import time
import numpy as np

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=False)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def build_compressed_checkpoint_name(checkpoint_file, prune_ratio, quantize):
    base, ext = os.path.splitext(checkpoint_file)
    suffix = []
    if prune_ratio > 0:
        suffix.append(f'pruned{int(prune_ratio*100)}')
    if quantize:
        suffix.append('int8')
    return base + '_' + '_'.join(suffix) + ext if suffix else checkpoint_file


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # Remove weight norm before pruning / quantization / inference
    generator.remove_weight_norm()

    if a.prune_ratio > 0:
        print(f"Applying structured pruning with ratio {a.prune_ratio:.2f}...")
        prune_conv_layers(generator, a.prune_ratio)

    if a.quantize:
        if device.type != 'cpu':
            print('Quantized inference uses CPU. Moving model to CPU for INT8 dynamic quantization.')
        generator = generator.to('cpu')
        generator = quantize_dynamic_model(generator)

    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()

    loaded_size = file_size_in_mb(a.checkpoint_file)
    num_params = count_parameters(generator)
    print(f"Loaded checkpoint: {a.checkpoint_file} ({loaded_size:.3f} MB)")
    print(f"Generator parameters: {num_params:,}")

    if (a.prune_ratio > 0 or a.quantize) and a.save_compressed_checkpoint:
        compressed_path = a.compressed_checkpoint_file or build_compressed_checkpoint_name(a.checkpoint_file, a.prune_ratio, a.quantize)
        save_checkpoint(compressed_path, {'generator': generator.state_dict()})
        compressed_size = file_size_in_mb(compressed_path)
        print(f"Saved compressed checkpoint: {compressed_path} ({compressed_size:.3f} MB)")

    total_inference_time = 0
    total_audio_duration = 0

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)

            start_time = time.time()

            x = get_mel(wav.unsqueeze(0)).to(device)
            y_g_hat = generator(x)

            inference_time = time.time() - start_time

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            audio_duration = len(wav) / h.sampling_rate
            rtf = inference_time / audio_duration

            total_inference_time += inference_time
            total_audio_duration += audio_duration

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(f"{output_file} | RTF: {rtf:.4f}({inference_time:.3f}s / {audio_duration:.3f}s)")
        
        avg_rtf = total_inference_time / total_audio_duration
        print(f"Average RTF: {avg_rtf:.4f}({total_inference_time:.3f}s / {total_audio_duration:.3f}s)")
    
    # Evaluate generated files against references using functions from utils.py
    pairs = collect_pairs(a.input_wavs_dir, a.output_dir, a.generated_suffix if hasattr(a, 'generated_suffix') else '_generated')
    mel_l1_scores = []
    pesq_scores = []
    stoi_scores = []

    for (ref_path, gen_path) in pairs:
        ref_sr, ref_audio = load_audio(ref_path)
        gen_sr, gen_audio = load_audio(gen_path)

        ref_audio, gen_audio = align_waveforms(ref_audio, gen_audio)

        ref_mel = compute_mel(ref_audio, h)
        gen_mel = compute_mel(gen_audio, h)
        mel_l1 = torch.nn.functional.l1_loss(ref_mel, gen_mel).item()
        mel_l1_scores.append(mel_l1)

        ref_16k = resample_audio(ref_audio, h.sampling_rate, 16000)
        gen_16k = resample_audio(gen_audio, h.sampling_rate, 16000)
        ref_16k, gen_16k = align_waveforms(ref_16k, gen_16k)

        pesq_scores.append(pesq_fn(16000, ref_16k, gen_16k, "wb"))
        stoi_scores.append(stoi_fn(ref_16k, gen_16k, 16000, extended=False))

    mean_mel_l1 = float(np.mean(mel_l1_scores)) if mel_l1_scores else None
    mean_pesq = float(np.mean(pesq_scores)) if pesq_scores else None
    mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else None

    # Print evaluation summary
    print("Evaluation summary:")
    print(f"  Mel-spectrogram L1: {mean_mel_l1:.6f}" if mean_mel_l1 is not None else "  Mel-spectrogram L1: None")
    print(f"  PESQ: {mean_pesq:.6f}" if mean_pesq is not None else "  PESQ: None")
    print(f"  STOI: {mean_stoi:.6f}" if mean_stoi is not None else "  STOI: None")

    metrics = {
        'experiment_name': a.experiment_name if hasattr(a, 'experiment_name') and a.experiment_name else os.path.splitext(os.path.basename(a.checkpoint_file))[0],
        'num_params': num_params,
        'model_size_mb': round(loaded_size, 3),
        'avg_rtf': round(avg_rtf, 4),
        'pesq': mean_pesq,
        'stoi': mean_stoi,
        'mel_l1': mean_mel_l1,
    }

    # Write to CSV if requested
    if hasattr(a, 'csv_file') and a.csv_file:
        csv_path = a.csv_file
        header = ['experiment_name','num_params','model_size_mb','avg_rtf','pesq','stoi','mel_l1']
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'experiment_name': metrics['experiment_name'],
                'num_params': metrics['num_params'],
                'model_size_mb': metrics['model_size_mb'],
                'avg_rtf': metrics['avg_rtf'],
                'pesq': metrics['pesq'],
                'stoi': metrics['stoi'],
                'mel_l1': metrics['mel_l1'],
            })
        print(f"\nAppended metrics to CSV: {csv_path}")

    return metrics


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='LibriSpeech_wav/test')
    parser.add_argument('--output_dir', default='generated_audios/generated_files_LibriSpeech_wav') # baseline: generated_files_LibriSpeech_wav
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--quantize', action='store_true', help='Apply INT8 dynamic quantization to the generator')
    parser.add_argument('--prune_ratio', default=0.0, type=float,
                        help='Structured pruning ratio for Conv1d and ConvTranspose1d weights')
    parser.add_argument('--save_compressed_checkpoint', action='store_true',
                        help='Save the pruned / quantized model checkpoint')
    parser.add_argument('--compressed_checkpoint_file', default=None,
                        help='Optional path to save the compressed checkpoint')
    parser.add_argument('--experiment_name', default=None,
                        help='Experiment name to record in CSV')
    parser.add_argument('--csv_file', default=None,
                        help='Optional CSV file to append metrics to')
    parser.add_argument('--generated_suffix', default='_generated',
                        help='Suffix used for generated files (default: _generated)')
    a = parser.parse_args()

    config_file = a.config_file
    # config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    device = torch.device('cpu')
    print('Using device: cpu (baseline and quantized inference both run on CPU)')
    if a.quantize:
        print('Quantization enabled: forcing CPU inference for INT8 model.')

    inference(a)


if __name__ == '__main__':
    main()

