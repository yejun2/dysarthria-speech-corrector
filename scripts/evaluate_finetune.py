#!/usr/bin/env python3
"""Evaluate an Allosaurus model against a wave/text manifest using PER."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import editdistance

from allosaurus.app import read_recognizer


def read_manifest(manifest_dir: Path) -> list[tuple[str, str, list[str]]]:
    wavs: dict[str, str] = {}
    references: dict[str, list[str]] = {}
    for line in (manifest_dir / "wave").read_text(encoding="utf-8").splitlines():
        utterance_id, path = line.strip().split(maxsplit=1)
        wavs[utterance_id] = path
    for line in (manifest_dir / "text").read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if fields:
            references[fields[0]] = fields[1:]
    if set(wavs) != set(references):
        missing_text = sorted(set(wavs) - set(references))
        missing_wave = sorted(set(references) - set(wavs))
        raise ValueError(f"manifest ID mismatch; missing text={missing_text[:5]}, missing wave={missing_wave[:5]}")
    return [(uid, wavs[uid], references[uid]) for uid in sorted(wavs)]


def evaluate(model: str, lang: str, manifest_dir: Path, output_dir: Path, device_id: int) -> dict:
    examples = read_manifest(manifest_dir)
    recognizer = read_recognizer(
        Namespace(model=model, device_id=device_id, lang=lang, approximate=False, prior=None)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    total_edits = 0
    total_reference_phones = 0
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for uid, wav_path, reference in examples:
            prediction_text = recognizer.recognize(wav_path, lang)
            prediction = prediction_text.split()
            edits = editdistance.distance(reference, prediction)
            total_edits += edits
            total_reference_phones += len(reference)
            stream.write(
                json.dumps(
                    {
                        "utterance_id": uid,
                        "wav_path": wav_path,
                        "reference": reference,
                        "prediction": prediction,
                        "edit_distance": edits,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if total_reference_phones == 0:
        raise ValueError("test manifest contains no reference phones")
    metrics = {
        "model": model,
        "lang": lang,
        "utterances": len(examples),
        "reference_phones": total_reference_phones,
        "edit_distance": total_edits,
        "phone_error_rate": total_edits / total_reference_phones,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute phone error rate for an Allosaurus model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", default="kor")
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=-1)
    args = parser.parse_args()
    try:
        metrics = evaluate(args.model, args.lang, args.manifest_dir, args.output_dir, args.device_id)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
