from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from torch.ao import quantization
try:
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
except ImportError:
    prepare_fx = None
    convert_fx = None
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
from utils import (
    count_parameters,
    file_size_in_mb,
    prune_conv_layers,
    prune_conv_layers_with_mask,
    remove_prune_masks,
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



def append_per_audio_metrics_to_csv(csv_path, rows, header):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def append_metrics_to_csv(csv_path, metrics, header):
    existing_rows = []
    existing_header = None
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as cf:
            reader = csv.DictReader(cf)
            existing_header = reader.fieldnames
            existing_rows = list(reader)

    if existing_header:
        merged_header = list(existing_header)
        for field in header:
            if field not in merged_header:
                merged_header.append(field)
    else:
        merged_header = header

    row = {field: metrics.get(field) for field in merged_header}
    write_mode = 'w' if existing_header != merged_header else 'a'
    with open(csv_path, write_mode, newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=merged_header)
        if write_mode == 'w':
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
        writer.writerow(row)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=False)
    print("Complete.")
    return checkpoint_dict


def is_masked_checkpoint(state_dict):
    return any(k.endswith('weight_mask') or k.endswith('weight_orig') for k in state_dict.keys())


def is_weight_norm_checkpoint(state_dict):
    return any(k.endswith('weight_g') or k.endswith('weight_v') for k in state_dict.keys())


def prepare_generator_for_checkpoint(generator, state_dict, prune_convtranspose=False):
    if is_masked_checkpoint(state_dict):
        print('Checkpoint contains masked pruning parameters. Preparing masked model before loading...')
        generator.remove_weight_norm()
        prune_conv_layers_with_mask(generator, amount=0.0, prune_convtranspose=prune_convtranspose)
        generator.load_state_dict(state_dict)
        return 'masked'

    if is_weight_norm_checkpoint(state_dict):
        print('Checkpoint contains weight_norm parameters. Keeping weight_norm wrappers for loading...')
        generator.load_state_dict(state_dict)
        return 'weight_norm'

    print('Checkpoint contains plain conv weights. Removing weight_norm wrappers before loading...')
    generator.remove_weight_norm()
    generator.load_state_dict(state_dict)
    return 'plain'


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def print_quantized_modules(model):
    quantized_found = False
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_path = type(module).__module__
        if 'quantized' in module_path or 'Quantized' in module_type:
            print(f"Quantized module: {name} ({module_path}.{module_type})")
            quantized_found = True
    if not quantized_found:
        print("No quantized modules found in model.")
    return quantized_found


def _select_quantized_engine():
    supported = torch.backends.quantized.supported_engines
    for engine in ('fbgemm', 'x86', 'qnnpack'):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No supported quantized engine found. Available engines: {supported}")


def static_quantize_model(generator, calibration_inputs, quantize_scope='all',
                          quantize_resblock_start=3, quantize_resblock_end=8,
                          dtype=torch.qint8):
    if prepare_fx is None or convert_fx is None:
        raise RuntimeError(
            "FX graph mode quantization is not available in this PyTorch build. "
            "Conv1d eager static quantization needs QuantStub/DeQuantStub in the model, "
            "so this script uses FX quantization to avoid editing models.py.")
    if not calibration_inputs:
        raise RuntimeError("Need at least one calibration input for static INT8 quantization.")

    engine = _select_quantized_engine()
    generator.eval()

    convtranspose_qconfig = quantization.QConfig(
        activation=quantization.default_observer,
        weight=quantization.default_weight_observer,
    )
    default_qconfig = quantization.get_default_qconfig(engine)
    if hasattr(quantization, 'get_default_qconfig_mapping'):
        qconfig_mapping = quantization.get_default_qconfig_mapping(engine)
        qconfig_mapping = qconfig_mapping.set_object_type(
            torch.nn.ConvTranspose1d, convtranspose_qconfig)
    else:
        qconfig_mapping = quantization.QConfigMapping().set_global(convtranspose_qconfig)

    if quantize_scope == 'resblocks_range':
        if quantize_resblock_start < 0 or quantize_resblock_end < quantize_resblock_start:
            raise ValueError(
                'Expected 0 <= quantize_resblock_start <= quantize_resblock_end, '
                f'got {quantize_resblock_start}..{quantize_resblock_end}')
        block_ids = '|'.join(str(i) for i in range(quantize_resblock_start, quantize_resblock_end + 1))
        qconfig_mapping = quantization.QConfigMapping().set_global(None)
        qconfig_mapping = qconfig_mapping.set_module_name_regex(
            rf'resblocks\.({block_ids})\..*', default_qconfig)
    else:
        if quantize_scope in ('no_upsample', 'resblocks'):
            qconfig_mapping = qconfig_mapping.set_object_type(torch.nn.ConvTranspose1d, None)
        if quantize_scope == 'resblocks':
            qconfig_mapping = qconfig_mapping.set_module_name('conv_pre', None)
            qconfig_mapping = qconfig_mapping.set_module_name('conv_post', None)
        elif quantize_scope == 'no_output':
            qconfig_mapping = qconfig_mapping.set_module_name('conv_post', None)

    print(f"INT8 quantization scope: {quantize_scope}")
    if quantize_scope == 'resblocks_range':
        print(
            f"Quantizing residual blocks {quantize_resblock_start}..{quantize_resblock_end}; "
            "all other layers stay FP32.")
    example_inputs = (calibration_inputs[0],)
    try:
        prepared = prepare_fx(generator, qconfig_mapping, example_inputs)
        with torch.no_grad():
            for x in calibration_inputs:
                prepared(x)
        return convert_fx(prepared)
    except Exception as exc:
        raise RuntimeError(
            "FX INT8 quantization failed for this HiFi-GAN Generator. "
            "Do not use eager Conv1d static quantization here: it converts conv_pre "
            "without adding a quantized input boundary, which causes the "
            "quantized::conv1d CPU-backend error seen in the log. "
            f"Original error: {type(exc).__name__}: {exc}") from exc


def build_compressed_checkpoint_name(checkpoint_file, prune_ratio, quantize, clean=False):
    base, ext = os.path.splitext(checkpoint_file)
    suffix = []
    if clean:
        suffix.append('clean')
    if prune_ratio > 0:
        suffix.append(f'pruned{int(prune_ratio*100)}')
    if quantize:
        suffix.append('int8')
    return base + '_' + '_'.join(suffix) + ext if suffix else checkpoint_file


def count_state_dict_parameters(state_dict):
    return sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def _effective_pruned_weight(module):
    weight = getattr(module, 'weight_orig', module.weight)
    mask = getattr(module, 'weight_mask', None)
    if mask is not None:
        weight = weight * mask
    return weight.detach()


def _copy_conv1d_like(old_conv, in_channels, out_channels):
    return torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    ).to(next(old_conv.parameters()).device)


def structurally_compact_masked_resblocks(generator, min_channels=1):
    """Physically remove masked middle channels inside ResBlock1 conv pairs.

    Each pair is converted from C -> C -> C to C -> K -> C. The residual
    block output channel count stays unchanged, so skip connections still match.
    This is an inference-time compaction for masked fine-tuned checkpoints.
    """
    total_removed = 0
    total_before = 0
    total_after = 0
    compacted_pairs = 0

    for block_name, block in generator.named_modules():
        if not (hasattr(block, 'convs1') and hasattr(block, 'convs2')):
            continue

        for idx, (old_c1, old_c2) in enumerate(zip(block.convs1, block.convs2)):
            if not hasattr(old_c1, 'weight_mask'):
                continue

            c1_weight = _effective_pruned_weight(old_c1)
            c2_weight = _effective_pruned_weight(old_c2)
            c1_mask = old_c1.weight_mask.detach()
            keep = c1_mask.reshape(c1_mask.shape[0], -1).sum(dim=1) > 0

            if int(keep.sum().item()) < min_channels:
                scores = c1_weight.pow(2).sum(dim=(1, 2))
                keep_idx = torch.topk(scores, k=min(min_channels, scores.numel())).indices.sort().values
            else:
                keep_idx = torch.nonzero(keep, as_tuple=False).flatten()

            kept_channels = int(keep_idx.numel())
            old_params = old_c1.weight.numel() + old_c2.weight.numel()
            if old_c1.bias is not None:
                old_params += old_c1.bias.numel()
            if old_c2.bias is not None:
                old_params += old_c2.bias.numel()

            new_c1 = _copy_conv1d_like(old_c1, old_c1.in_channels, kept_channels)
            new_c2 = _copy_conv1d_like(old_c2, kept_channels, old_c2.out_channels)
            new_c1 = new_c1.to(dtype=c1_weight.dtype)
            new_c2 = new_c2.to(dtype=c2_weight.dtype)

            with torch.no_grad():
                new_c1.weight.copy_(c1_weight.index_select(0, keep_idx))
                if old_c1.bias is not None:
                    new_c1.bias.copy_(old_c1.bias.detach().index_select(0, keep_idx))
                new_c2.weight.copy_(c2_weight.index_select(1, keep_idx))
                if old_c2.bias is not None:
                    new_c2.bias.copy_(old_c2.bias.detach())

            block.convs1[idx] = new_c1
            block.convs2[idx] = new_c2

            new_params = new_c1.weight.numel() + new_c2.weight.numel()
            if new_c1.bias is not None:
                new_params += new_c1.bias.numel()
            if new_c2.bias is not None:
                new_params += new_c2.bias.numel()

            total_before += old_params
            total_after += new_params
            total_removed += old_params - new_params
            compacted_pairs += 1
            print(
                f"Compacted {block_name}.convs pair {idx}: "
                f"hidden channels {old_c1.out_channels} -> {kept_channels}")

    print(
        f"Structural mask compaction: {compacted_pairs} conv pairs, "
        f"removed {total_removed:,} / {total_before:,} pair parameters")
    return {
        'compacted_pairs': compacted_pairs,
        'pair_params_before': total_before,
        'pair_params_after': total_after,
        'pair_params_removed': total_removed,
    }


def inference(a):
    generator = Generator(h).to(device)

    checkpoint = load_checkpoint(a.checkpoint_file, device)
    state_dict_g = checkpoint['generator'] if 'generator' in checkpoint else checkpoint

    checkpoint_type = prepare_generator_for_checkpoint(generator, state_dict_g, prune_convtranspose=a.prune_convtranspose)

    if checkpoint_type == 'masked':
        if a.structural_prune_masks:
            print('Physically compacting masked residual channels before inference...')
            structurally_compact_masked_resblocks(
                generator, min_channels=a.structural_prune_min_channels)
        else:
            print('Removing pruning masks from masked checkpoint before inference...')
            remove_prune_masks(generator)
    elif a.prune_ratio > 0:
        print(f'Applying structured pruning with ratio {a.prune_ratio:.2f}...')
        prune_conv_layers(generator, a.prune_ratio, prune_convtranspose=a.prune_convtranspose)

    filelist = sorted(os.listdir(a.input_wavs_dir))
    os.makedirs(a.output_dir, exist_ok=True)

    if a.quantize:
        if device.type != 'cpu':
            print('Quantized inference uses CPU. Moving model to CPU for INT8 static quantization.')
        generator.remove_weight_norm()  # Quantization works better without weight_norm wrappers
        generator = generator.to('cpu')

        calibration_files = filelist[:min(len(filelist), a.calibration_samples)]
        calibration_inputs = []
        for filename in calibration_files:
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).unsqueeze(0).to('cpu')
            x = get_mel(wav).to('cpu')
            calibration_inputs.append(x)

        print(f'Calibrating static quantization using {len(calibration_inputs)} samples...')
        generator = static_quantize_model(
            generator, calibration_inputs,
            quantize_scope=a.quantize_scope,
            quantize_resblock_start=a.quantize_resblock_start,
            quantize_resblock_end=a.quantize_resblock_end)
        print('Checking quantized modules after static quantization:')
        print_quantized_modules(generator)

    generator.eval()

    loaded_size = file_size_in_mb(a.checkpoint_file)
    should_save_compressed = a.save_compressed_checkpoint and (
        a.prune_ratio > 0 or a.quantize or checkpoint_type == 'masked'
    )
    compressed_path = None
    compressed_size = None
    compressed_num_params = None
    if should_save_compressed:
        clean_suffix = checkpoint_type == 'masked' and a.prune_ratio == 0 and not a.quantize
        compressed_path = a.compressed_checkpoint_file or build_compressed_checkpoint_name(
            a.checkpoint_file, a.prune_ratio, a.quantize, clean=clean_suffix)
        save_checkpoint(compressed_path, {'generator': generator.state_dict()})
        if checkpoint_type == 'masked' and a.structural_prune_masks:
            print('Note: structurally compacted checkpoints cannot be reloaded by the standard Generator without a matching structural config.')
        compressed_size = file_size_in_mb(compressed_path)
        compressed_num_params = count_state_dict_parameters(generator.state_dict())
        print(f"Saved compressed checkpoint: {compressed_path} ({compressed_size:.3f} MB)")
        print(f"Compressed checkpoint parameters: {compressed_num_params:,}")
        if clean_suffix:
            print(f"Clean mask-free checkpoint saved: {compressed_path} ({compressed_size:.3f} MB)")

    num_params = compressed_num_params if compressed_num_params is not None else count_parameters(generator)
    print(f"Loaded checkpoint: {a.checkpoint_file} ({loaded_size:.3f} MB)")
    print(f"Generator parameters: {num_params:,}")

    total_inference_time = 0
    total_generator_time = 0
    total_audio_duration = 0
    per_audio_metrics = {}

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)

            start_time = time.perf_counter()
            x = get_mel(wav.unsqueeze(0)).to(device)

            generator_start_time = time.perf_counter()
            y_g_hat = generator(x)
            generator_time = time.perf_counter() - generator_start_time

            inference_time = time.perf_counter() - start_time

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            audio_duration = len(wav) / h.sampling_rate
            rtf = inference_time / audio_duration
            generator_rtf = generator_time / audio_duration

            total_inference_time += inference_time
            total_generator_time += generator_time
            total_audio_duration += audio_duration

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + a.generated_suffix + '.wav')
            per_audio_metrics[os.path.abspath(output_file)] = {
                'audio_file': filname,
                'generated_file': output_file,
                'audio_duration': audio_duration,
                'inference_time': inference_time,
                'generator_time': generator_time,
                'avg_rtf': rtf,
                'avg_generator_rtf': generator_rtf,
            }
            write(output_file, h.sampling_rate, audio)
            print(
                f"{output_file} | RTF: {rtf:.4f}({inference_time:.3f}s / {audio_duration:.3f}s) | "
                f"Generator RTF: {generator_rtf:.4f}({generator_time:.3f}s / {audio_duration:.3f}s)")

        avg_rtf = total_inference_time / total_audio_duration
        avg_generator_rtf = total_generator_time / total_audio_duration
        print(f"Average RTF: {avg_rtf:.4f}({total_inference_time:.3f}s / {total_audio_duration:.3f}s)")
        print(
            f"Average Generator RTF: {avg_generator_rtf:.4f}"
            f"({total_generator_time:.3f}s / {total_audio_duration:.3f}s)")
    
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

        pesq = pesq_fn(16000, ref_16k, gen_16k, "wb")
        stoi = stoi_fn(ref_16k, gen_16k, 16000, extended=False)
        pesq_scores.append(pesq)
        stoi_scores.append(stoi)

        row = per_audio_metrics.get(os.path.abspath(str(gen_path)))
        if row is not None:
            row['pesq'] = pesq
            row['stoi'] = stoi
            row['mel_l1'] = mel_l1

    mean_mel_l1 = float(np.mean(mel_l1_scores)) if mel_l1_scores else None
    mean_pesq = float(np.mean(pesq_scores)) if pesq_scores else None
    mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else None

    # Print evaluation summary
    print("Evaluation summary:")
    print(f"  Mel-spectrogram L1: {mean_mel_l1:.6f}" if mean_mel_l1 is not None else "  Mel-spectrogram L1: None")
    print(f"  PESQ: {mean_pesq:.6f}" if mean_pesq is not None else "  PESQ: None")
    print(f"  STOI: {mean_stoi:.6f}" if mean_stoi is not None else "  STOI: None")

    display_size = compressed_size if compressed_size is not None else loaded_size
    metrics = {
        'experiment_name': a.experiment_name if hasattr(a, 'experiment_name') and a.experiment_name else os.path.splitext(os.path.basename(a.checkpoint_file))[0],
        'num_params': num_params,
        'model_size_mb': round(display_size, 3),
        'avg_rtf': round(avg_rtf, 4),
        'avg_generator_rtf': round(avg_generator_rtf, 4),
        'pesq': mean_pesq,
        'stoi': mean_stoi,
        'mel_l1': mean_mel_l1,
    }

    per_audio_rows = []
    experiment_name = metrics['experiment_name']
    for row in per_audio_metrics.values():
        row = dict(row)
        row.setdefault('pesq', None)
        row.setdefault('stoi', None)
        row.setdefault('mel_l1', None)
        row['experiment_name'] = experiment_name
        for key in ('audio_duration', 'inference_time', 'generator_time', 'avg_rtf', 'avg_generator_rtf', 'pesq', 'stoi', 'mel_l1'):
            if row[key] is not None:
                row[key] = round(float(row[key]), 6)
        per_audio_rows.append(row)

    if hasattr(a, 'per_audio_csv_file') and a.per_audio_csv_file:
        per_audio_header = [
            'experiment_name', 'audio_file', 'generated_file',
            'audio_duration', 'inference_time', 'generator_time',
            'avg_rtf', 'avg_generator_rtf', 'pesq', 'stoi', 'mel_l1']
        append_per_audio_metrics_to_csv(a.per_audio_csv_file, per_audio_rows, per_audio_header)
        print(f"Appended per-audio metrics to CSV: {a.per_audio_csv_file}")

    # Write to CSV if requested
    if hasattr(a, 'csv_file') and a.csv_file:
        csv_path = a.csv_file
        header = [
            'experiment_name', 'num_params', 'model_size_mb', 'avg_rtf',
            'avg_generator_rtf', 'pesq', 'stoi', 'mel_l1']
        append_metrics_to_csv(csv_path, metrics, header)
        print(f"\nAppended metrics to CSV: {csv_path}")

    return metrics


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='LibriSpeech_wav/test')
    parser.add_argument('--output_dir', default='generated_audios/generated_files_LibriSpeech_wav') # baseline: generated_files_LibriSpeech_wav
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--quantize', action='store_true', help='Apply INT8 quantization to the generator')
    parser.add_argument('--quantize_scope', default='resblocks_range',
                        choices=['all', 'no_output', 'no_upsample', 'resblocks', 'resblocks_range'],
                        help=("Quantization scope:\n"
                                " all             : Quantize all layers.\n"
                                " no_output       : Keep conv_post in FP32.\n"
                                " no_upsample     : Keep ConvTranspose1d upsampling layers in FP32.\n"
                                " resblocks       : Quantize all residual blocks; keep conv_pre, ConvTranspose1d and conv_post in FP32.\n"
                                " resblocks_range : Quantize only selected residual blocks; all other layers stay FP32."))
    parser.add_argument('--quantize_resblock_start', default=3, type=int,
                        help='First residual block index for --quantize_scope resblocks_range')
    parser.add_argument('--quantize_resblock_end', default=8, type=int,
                        help='Last residual block index for --quantize_scope resblocks_range')
    parser.add_argument('--calibration_samples', default=50, type=int,
                        help='Number of input wavs used to calibrate static INT8 quantization')
    parser.add_argument('--prune_ratio', default=0.0, type=float,
                        help='Structured pruning ratio for Conv1d weights (ConvTranspose1d pruning is disabled by default)')
    parser.add_argument('--prune_convtranspose', action='store_true',
                        help='Also prune ConvTranspose1d output channels when using --prune_ratio')
    parser.add_argument('--structural_prune_masks', action='store_true',
                        help='For masked checkpoints, physically compact ResBlock1 middle channels in memory')
    parser.add_argument('--structural_prune_min_channels', default=1, type=int,
                        help='Minimum hidden channels kept per compacted residual conv pair')
    parser.add_argument('--save_compressed_checkpoint', action='store_true',
                        help='Save the pruned / quantized model checkpoint')
    parser.add_argument('--compressed_checkpoint_file', default=None,
                        help='Optional path to save the compressed checkpoint')
    parser.add_argument('--experiment_name', default=None,
                        help='Experiment name to record in CSV')
    parser.add_argument('--csv_file', default=None,
                        help='Optional CSV file to append experiment-level metrics to')
    parser.add_argument('--per_audio_csv_file', default=None,
                        help='Optional CSV file to save per-audio metrics for boxplots and p-value tests')
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

