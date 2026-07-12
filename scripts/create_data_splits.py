#!/usr/bin/env python3
"""Create fixed-size Allosaurus split manifests without changing source data."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

try:
    from scripts.audit_finetune_data import relative_key, utterance_id
    from scripts.prepare_finetune_data import Item, split_items_fixed, write_manifests
except ModuleNotFoundError:  # Support direct script execution.
    from audit_finetune_data import relative_key, utterance_id
    from prepare_finetune_data import Item, split_items_fixed, write_manifests


def collect_pairs(data_root: Path) -> tuple[list[Item], Counter[str]]:
    wavs = {
        relative_key(path, data_root): path
        for path in sorted(data_root.rglob("*.wav"))
        if path.is_file()
    }
    labels = {
        relative_key(path, data_root): path
        for path in sorted(data_root.rglob("*.txt"))
        if path.is_file()
    }
    if set(wavs) != set(labels):
        missing_labels = sorted(set(wavs) - set(labels))
        missing_wavs = sorted(set(labels) - set(wavs))
        raise ValueError(
            f"WAV/TXT mismatch: missing labels={len(missing_labels)}, "
            f"missing WAVs={len(missing_wavs)}"
        )

    items: list[Item] = []
    phone_frequencies: Counter[str] = Counter()
    for key in sorted(wavs):
        phones = tuple(labels[key].read_text(encoding="utf-8").strip().split())
        if not phones:
            raise ValueError(f"empty IPA label: {labels[key]}")
        phone_frequencies.update(phones)
        items.append(
            Item(
                utterance_id=utterance_id(key),
                relative_key=key,
                wav_path=str(wavs[key].resolve()),
                label_path=str(labels[key].resolve()),
                phones=phones,
            )
        )
    return items, phone_frequencies


def create_splits(
    data_root: Path,
    output_dir: Path,
    validate_count: int,
    test_count: int,
    seed: int,
    overwrite: bool = False,
) -> dict:
    items, phone_frequencies = collect_pairs(data_root)
    splits = split_items_fixed(items, validate_count, test_count, seed)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    write_manifests(splits, output_dir)

    ids = {name: {item.utterance_id for item in values} for name, values in splits.items()}
    if ids["train"] & ids["validate"] or ids["train"] & ids["test"] or ids["validate"] & ids["test"]:
        raise RuntimeError("split overlap detected")

    summary = {
        "data_root": str(data_root.resolve()),
        "seed": seed,
        "counts": {name: len(values) for name, values in splits.items()},
        "total": len(items),
        "unique_phones": len(phone_frequencies),
        "phone_frequencies": dict(sorted(phone_frequencies.items())),
        "overlap": {"train_validate": 0, "train_test": 0, "validate_test": 0},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create exact-size train/validate/test manifests")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-count", type=int, default=5000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"data root is not a directory: {args.data_root}")
    try:
        summary = create_splits(
            args.data_root,
            args.output_dir,
            args.validate_count,
            args.test_count,
            args.seed,
            args.overwrite,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
