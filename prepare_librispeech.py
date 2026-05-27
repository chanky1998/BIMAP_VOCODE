import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

VALIDATION_SAMPLE_SIZE = 500
RANDOM_SEED = 1234


SPLIT_CONFIG = {
    "train": {
        "source": Path("LibriSpeech/Train/train-clean-100"),
        "output": "train",
        "list_file": "training.txt",
    },
    "dev": {
        "source": Path("LibriSpeech/Dev/dev-clean"),
        "output": "dev",
        "list_file": "validation.txt",
    },
    "test": {
        "source": Path("LibriSpeech/Test/test-clean"),
        "output": "test",
        "list_file": "test.txt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flatten LibriSpeech splits into wav directories for HiFi-GAN."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root containing the LibriSpeech directory. Default: current directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="LibriSpeech_wav",
        help="Output directory for flattened wavs and file lists.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_CONFIG.keys()),
        default=["train", "dev", "test"],
        help="Which LibriSpeech splits to process.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target wav sample rate. Default: 22050.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        default=True,
        help="Convert audio to mono. Enabled by default.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip wavs that are already present in the output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files per split for quick testing.",
    )
    return parser.parse_args()


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install it with "
            "`conda install -c conda-forge ffmpeg` or `brew install ffmpeg`."
        )


def find_flacs(source_dir: Path):
    return sorted(p for p in source_dir.rglob("*.flac") if p.is_file())


def convert_flac_to_wav(src: Path, dst: Path, sample_rate: int, mono: bool):
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
    ]
    if mono:
        command.extend(["-ac", "1"])
    command.append(str(dst))
    subprocess.run(command, check=True)


def write_list_file(list_path: Path, wav_names):
    list_path.parent.mkdir(parents=True, exist_ok=True)
    split_name = list_path.stem
    if split_name == "training":
        tag = "LibriSpeech-train"
    elif split_name == "validation":
        tag = "LibriSpeech-dev"
    elif split_name == "test":
        tag = "LibriSpeech-test"
    else:
        tag = "LibriSpeech"
    if split_name == "validation" and len(wav_names) > VALIDATION_SAMPLE_SIZE:
        rng = random.Random(RANDOM_SEED)
        wav_names = sorted(rng.sample(wav_names, VALIDATION_SAMPLE_SIZE))
    content = "".join(f"{name}|{tag}\n" for name in wav_names)
    list_path.write_text(content, encoding="utf-8")


def process_split(root: Path, out_dir: Path, split: str, sample_rate: int, mono: bool,
                  skip_existing: bool, limit: Optional[int]):
    config = SPLIT_CONFIG[split]
    source_dir = root / config["source"]
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    wav_dir = out_dir / config["output"]
    wav_dir.mkdir(parents=True, exist_ok=True)

    flac_files = find_flacs(source_dir)
    if limit is not None:
        flac_files = flac_files[:limit]

    wav_names = []
    for index, src in enumerate(flac_files, start=1):
        stem = src.stem
        dst = wav_dir / f"{stem}.wav"
        wav_names.append(stem)

        if skip_existing and dst.exists():
            continue

        convert_flac_to_wav(src, dst, sample_rate, mono)

        if index % 100 == 0 or index == len(flac_files):
            print(f"[{split}] processed {index}/{len(flac_files)}")

    write_list_file(out_dir / config["list_file"], wav_names)
    print(f"[{split}] wavs -> {wav_dir}")
    print(f"[{split}] list -> {out_dir / config['list_file']}")


def main():
    args = parse_args()
    require_ffmpeg()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        process_split(
            root=root,
            out_dir=out_dir,
            split=split,
            sample_rate=args.sample_rate,
            mono=args.mono,
            skip_existing=args.skip_existing,
            limit=args.limit,
        )

    print("Done.")
    print(f"Output directory: {out_dir}")
    print("Note: train.py expects one --input_wavs_dir, so use one split at a time or unify wav folders.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
