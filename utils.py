import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import math
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly
from pathlib import Path
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn
from meldataset import MAX_WAV_VALUE, mel_spectrogram


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def file_size_in_mb(filepath):
    return os.path.getsize(filepath) / 1024**2


def load_audio(path):
    sampling_rate, audio = read(path)
    audio = audio.astype(np.float32) / MAX_WAV_VALUE
    return sampling_rate, audio


def compute_mel(audio, h):
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    mel = mel_spectrogram(
        audio_tensor,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )
    return mel.squeeze(0)


def align_waveforms(reference, generated):
    min_len = min(len(reference), len(generated))
    return reference[:min_len], generated[:min_len]


def resample_audio(audio, source_sr, target_sr):
    if source_sr == target_sr:
        return audio
    gcd = math.gcd(source_sr, target_sr)
    up = target_sr // gcd
    down = source_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def collect_pairs(reference_dir, generated_dir, generated_suffix):
    pairs = []
    for ref_path in sorted(Path(reference_dir).glob("*.wav")):
        gen_name = ref_path.stem + generated_suffix + ".wav"
        gen_path = Path(generated_dir) / gen_name
        if gen_path.exists():
            pairs.append((ref_path, gen_path))
    return pairs


def prune_conv_layers(model, amount):
    """Apply structured output-channel pruning to Conv1d and ConvTranspose1d layers."""
    import torch.nn.utils.prune as prune

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            # dim=0 prunes whole output channels (structured channel pruning)
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')

    return model



def quantize_dynamic_model(model, dtype=torch.qint8):
    """Apply dynamic quantization to supported convolution layers."""
    model = model.to("cpu")
    model.eval()

    q_modules = {torch.nn.Conv1d}
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        q_modules,
        dtype=dtype
    )

    return quantized_model


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

