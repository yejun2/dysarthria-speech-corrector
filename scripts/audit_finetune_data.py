#!/usr/bin/env python3
"""Audit paired WAV and IPA label files before Allosaurus fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


KSPON_MARKERS = ("o/", "b/", "n/", "u/", "+", "(", ")", "/")


@dataclass
class Issue:
    severity: str
    code: str
    path: str
    message: str


@dataclass
class AudioInfo:
    utterance_id: str
    wav_path: str
    label_path: str
    sample_rate: int
    channels: int
    sample_width_bits: int
    frames: int
    duration_seconds: float
    token_count: int


def relative_key(path: Path, root: Path) -> str:
    """Return a stable pairing key, independent of file extension."""
    return path.relative_to(root).with_suffix("").as_posix()


def utterance_id(key: str) -> str:
    """Convert a relative path into an Allosaurus-safe utterance id."""
    return key.replace("/", "__").replace(" ", "_")


def index_files(root: Path, suffix: str) -> tuple[dict[str, Path], list[Issue]]:
    indexed: dict[str, Path] = {}
    issues: list[Issue] = []
    for path in sorted(root.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        key = relative_key(path, root)
        if key in indexed:
            issues.append(Issue("error", "duplicate_key", str(path), f"duplicate relative key: {key}"))
        else:
            indexed[key] = path
    return indexed, issues


def has_hangul(text: str) -> bool:
    return any(
        "HANGUL" in unicodedata.name(character, "")
        for character in text
        if not character.isspace()
    )


def inspect_label(path: Path) -> tuple[list[str], list[Issue]]:
    issues: list[Issue] = []
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        return [], [Issue("error", "label_not_utf8", str(path), str(exc))]
    except OSError as exc:
        return [], [Issue("error", "label_unreadable", str(path), str(exc))]

    if not text:
        return [], [Issue("error", "empty_label", str(path), "label is empty")]
    if "\n" in text or "\r" in text:
        issues.append(Issue("error", "multiline_label", str(path), "expected one IPA sequence per file"))
    if has_hangul(text):
        issues.append(Issue("error", "hangul_in_label", str(path), "label contains Hangul, not a space-separated IPA sequence"))
    markers = sorted({marker for marker in KSPON_MARKERS if marker in text})
    if markers:
        issues.append(Issue("error", "kspon_annotation", str(path), f"label contains KsponSpeech annotation markers: {', '.join(markers)}"))

    tokens = text.split()
    return tokens, issues


def inspect_wav(
    path: Path,
    expected_sample_rate: int | None,
    max_duration: float | None,
) -> tuple[dict[str, int | float] | None, list[Issue]]:
    issues: list[Issue] = []
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width_bits = audio.getsampwidth() * 8
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            compression = audio.getcomptype()
    except (wave.Error, EOFError, OSError) as exc:
        return None, [Issue("error", "invalid_wav", str(path), str(exc))]

    duration = frames / sample_rate if sample_rate else 0.0
    if compression != "NONE":
        issues.append(Issue("error", "compressed_wav", str(path), f"compression type is {compression}"))
    if channels != 1:
        issues.append(Issue("error", "not_mono", str(path), f"expected 1 channel, found {channels}"))
    if sample_rate <= 0 or frames <= 0:
        issues.append(Issue("error", "empty_audio", str(path), "audio has no usable samples"))
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        issues.append(Issue("warning", "sample_rate_mismatch", str(path), f"expected {expected_sample_rate} Hz, found {sample_rate} Hz"))
    if max_duration is not None and duration > max_duration:
        issues.append(Issue("warning", "audio_too_long", str(path), f"{duration:.2f}s exceeds {max_duration:.2f}s"))

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width_bits,
        "frames": frames,
        "duration_seconds": duration,
    }, issues


def audit(
    audio_root: Path,
    label_root: Path,
    expected_sample_rate: int | None = 16000,
    max_duration: float | None = 10.0,
) -> dict:
    wavs, issues = index_files(audio_root, ".wav")
    labels, label_index_issues = index_files(label_root, ".txt")
    issues.extend(label_index_issues)

    wav_keys, label_keys = set(wavs), set(labels)
    for key in sorted(wav_keys - label_keys):
        issues.append(Issue("error", "missing_label", str(wavs[key]), f"no matching .txt for {key}"))
    for key in sorted(label_keys - wav_keys):
        issues.append(Issue("error", "missing_wav", str(labels[key]), f"no matching .wav for {key}"))

    records: list[AudioInfo] = []
    token_counts: Counter[str] = Counter()
    ids: set[str] = set()
    for key in sorted(wav_keys & label_keys):
        uid = utterance_id(key)
        if uid in ids:
            issues.append(Issue("error", "duplicate_utterance_id", key, f"generated id is duplicated: {uid}"))
        ids.add(uid)

        tokens, label_issues = inspect_label(labels[key])
        audio, audio_issues = inspect_wav(wavs[key], expected_sample_rate, max_duration)
        issues.extend(label_issues)
        issues.extend(audio_issues)
        token_counts.update(tokens)
        if audio is not None:
            records.append(AudioInfo(uid, str(wavs[key].resolve()), str(labels[key].resolve()), token_count=len(tokens), **audio))

    severity_counts = Counter(issue.severity for issue in issues)
    return {
        "summary": {
            "wav_files": len(wavs),
            "label_files": len(labels),
            "paired_files": len(wav_keys & label_keys),
            "audited_audio_files": len(records),
            "total_duration_seconds": round(sum(record.duration_seconds for record in records), 3),
            "unique_tokens": len(token_counts),
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
        },
        "issues": [asdict(issue) for issue in issues],
        "token_frequencies": dict(sorted(token_counts.items())),
        "records": [asdict(record) for record in records],
    }


def print_report(report: dict, stream=sys.stdout) -> None:
    summary = report["summary"]
    print("Data audit summary", file=stream)
    for key, value in summary.items():
        print(f"  {key}: {value}", file=stream)
    if report["issues"]:
        print("Issues", file=stream)
        for issue in report["issues"]:
            print(f"  [{issue['severity'].upper()}] {issue['code']}: {issue['path']} - {issue['message']}", file=stream)


def existing_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paired WAV and IPA files for Allosaurus fine-tuning")
    parser.add_argument("root", nargs="?", type=existing_directory, help="root containing paired WAV/TXT files")
    parser.add_argument("--audio-root", type=existing_directory, help="WAV root when audio and labels are separate")
    parser.add_argument("--label-root", type=existing_directory, help="TXT root when audio and labels are separate")
    parser.add_argument("--expected-sample-rate", type=int, default=16000, help="expected sample rate; 0 disables check")
    parser.add_argument("--max-duration", type=float, default=10.0, help="warning threshold in seconds; 0 disables check")
    parser.add_argument("--json-output", type=Path, help="optional path for the complete JSON report")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.root is not None and (args.audio_root is not None or args.label_root is not None):
        parser.error("use either ROOT or --audio-root/--label-root")
    if args.root is None and (args.audio_root is None or args.label_root is None):
        parser.error("provide ROOT or both --audio-root and --label-root")

    audio_root = args.root or args.audio_root
    label_root = args.root or args.label_root
    report = audit(
        audio_root,
        label_root,
        expected_sample_rate=args.expected_sample_rate or None,
        max_duration=args.max_duration or None,
    )
    print_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
