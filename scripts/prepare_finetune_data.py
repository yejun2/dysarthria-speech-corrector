#!/usr/bin/env python3
"""Validate IPA labels and build reproducible Allosaurus data manifests."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from allosaurus.lm.inventory import Inventory
from allosaurus.model import get_model_path

try:
    from scripts.audit_finetune_data import audit, relative_key, utterance_id
except ModuleNotFoundError:  # Support: python scripts/prepare_finetune_data.py
    from audit_finetune_data import audit, relative_key, utterance_id


@dataclass(frozen=True)
class Item:
    utterance_id: str
    relative_key: str
    wav_path: str
    label_path: str
    phones: tuple[str, ...]


def load_phone_mapping(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {}
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and (isinstance(value, str) or value is None)
        for key, value in mapping.items()
    ):
        raise ValueError("phone mapping must be a JSON object of string -> string/null")
    return mapping


def apply_phone_mapping(tokens: list[str], mapping: dict[str, str | None]) -> tuple[str, ...]:
    mapped: list[str] = []
    for token in tokens:
        replacement = mapping.get(token, token)
        if replacement is None:
            continue
        mapped.extend(replacement.split())
    return tuple(unicodedata.normalize("NFC", token) for token in mapped)


def read_inventory(model: str, lang: str) -> set[str]:
    inventory = Inventory(get_model_path(model))
    inventory_path = Path(lang)
    if inventory_path.is_file():
        unit = inventory.get_unit(str(inventory_path))
    elif inventory.is_available(lang):
        unit = inventory.get_unit(lang)
    else:
        raise ValueError(f"language {lang!r} is not available in model {model!r}")
    return {
        unicodedata.normalize("NFC", phone)
        for phone in unit.unit_to_id
        if phone != "<blk>"
    }


def collect_items(
    data_root: Path,
    mapping: dict[str, str | None],
    inventory: set[str],
) -> tuple[list[Item], Counter[str], Counter[str]]:
    items: list[Item] = []
    frequencies: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    for wav_path in sorted(data_root.rglob("*.wav")):
        key = relative_key(wav_path, data_root)
        label_path = wav_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        raw_tokens = label_path.read_text(encoding="utf-8").strip().split()
        phones = apply_phone_mapping(raw_tokens, mapping)
        frequencies.update(phones)
        unsupported.update(phone for phone in phones if phone not in inventory)
        items.append(
            Item(
                utterance_id=utterance_id(key),
                relative_key=key,
                wav_path=str(wav_path.resolve()),
                label_path=str(label_path.resolve()),
                phones=phones,
            )
        )
    return items, frequencies, unsupported


def split_items(
    items: list[Item],
    train_ratio: float,
    validate_ratio: float,
    seed: int,
) -> dict[str, list[Item]]:
    if not 0 < train_ratio < 1 or not 0 < validate_ratio < 1 or train_ratio + validate_ratio >= 1:
        raise ValueError("ratios must be positive and train + validate must be below 1")
    if len(items) < 3:
        raise ValueError("at least 3 paired utterances are required for train/validate/test")

    shuffled = sorted(items, key=lambda item: item.relative_key)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    validate_count = max(1, round(total * validate_ratio))
    test_count = max(1, round(total * (1 - train_ratio - validate_ratio)))
    if validate_count + test_count >= total:
        validate_count = 1
        test_count = 1
    train_count = total - validate_count - test_count
    return {
        "train": shuffled[:train_count],
        "validate": shuffled[train_count : train_count + validate_count],
        "test": shuffled[train_count + validate_count :],
    }


def split_items_fixed(
    items: list[Item],
    validate_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[Item]]:
    """Split items reproducibly using exact validation and test counts."""
    if validate_count < 1 or test_count < 1:
        raise ValueError("validate_count and test_count must be positive")
    if validate_count + test_count >= len(items):
        raise ValueError("validate_count + test_count must be smaller than the dataset")

    shuffled = sorted(items, key=lambda item: item.relative_key)
    random.Random(seed).shuffle(shuffled)
    train_count = len(shuffled) - validate_count - test_count
    return {
        "train": shuffled[:train_count],
        "validate": shuffled[train_count : train_count + validate_count],
        "test": shuffled[train_count + validate_count :],
    }


def write_manifests(splits: dict[str, list[Item]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        wave_lines = [f"{item.utterance_id} {item.wav_path}" for item in items]
        text_lines = [f"{item.utterance_id} {' '.join(item.phones)}" for item in items]
        (split_dir / "wave").write_text("\n".join(wave_lines) + "\n", encoding="utf-8")
        (split_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def prepare(
    data_root: Path,
    output_dir: Path,
    model: str,
    lang: str,
    mapping_path: Path | None,
    train_ratio: float,
    validate_ratio: float,
    seed: int,
    expected_sample_rate: int | None,
    max_duration: float | None,
    overwrite: bool = False,
) -> dict:
    audit_report = audit(data_root, data_root, expected_sample_rate, max_duration)
    if audit_report["summary"]["errors"]:
        raise ValueError(f"data audit failed with {audit_report['summary']['errors']} error(s)")

    mapping = load_phone_mapping(mapping_path)
    inventory = read_inventory(model, lang)
    items, frequencies, unsupported = collect_items(data_root, mapping, inventory)
    if unsupported:
        detail = ", ".join(f"{phone}({count})" for phone, count in unsupported.most_common())
        raise ValueError(f"unsupported phones for {lang}/{model}: {detail}")
    if any(not item.phones for item in items):
        raise ValueError("phone mapping produced one or more empty labels")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    splits = split_items(items, train_ratio, validate_ratio, seed)
    write_manifests(splits, output_dir)
    summary = {
        "data_root": str(data_root.resolve()),
        "model": model,
        "lang": lang,
        "seed": seed,
        "train_ratio": train_ratio,
        "validate_ratio": validate_ratio,
        "counts": {name: len(values) for name, values in splits.items()},
        "unique_phones": len(frequencies),
        "phone_frequencies": dict(sorted(frequencies.items())),
        "audit": audit_report["summary"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train/validate/test manifests for Allosaurus")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="latest")
    parser.add_argument("--lang", default="kor")
    parser.add_argument("--phone-mapping", type=Path, help="optional JSON phone mapping")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validate-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"data root is not a directory: {args.data_root}")
    try:
        summary = prepare(
            args.data_root,
            args.output_dir,
            args.model,
            args.lang,
            args.phone_mapping,
            args.train_ratio,
            args.validate_ratio,
            args.seed,
            args.expected_sample_rate or None,
            args.max_duration or None,
            args.overwrite,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
