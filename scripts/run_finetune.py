#!/usr/bin/env python3
"""Run the initial Allosaurus fine-tuning pipeline by explicit stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from allosaurus.bin.prep_feat import prepare_feature
from allosaurus.bin.prep_token import prepare_token
from allosaurus.model import resolve_model_name

try:
    from scripts.evaluate_finetune import evaluate
    from scripts.prepare_finetune_data import prepare
except ModuleNotFoundError:  # Support: python scripts/run_finetune.py
    from evaluate_finetune import evaluate
    from prepare_finetune_data import prepare


ACOUSTIC_FEATURE_FILES = ("feat.scp", "feat.ark", "shape")
GENERATED_FEATURE_FILES = ACOUSTIC_FEATURE_FILES + ("token",)

CONFIG_SCHEMA = {
    "pipeline": {"stage": "stage"},
    "experiment": {"name": "wandb_run_name"},
    "data": {
        "data_root": "data_root",
        "work_dir": "work_dir",
        "phone_mapping": "phone_mapping",
        "train_ratio": "train_ratio",
        "validate_ratio": "validate_ratio",
        "seed": "seed",
        "expected_validate_size": "expected_validate_size",
    },
    "model": {
        "pretrained_model": "pretrained_model",
        "new_model": "new_model",
        "inventory": "lang",
        "phone_initialization": "phone_init_map",
    },
    "training": {
        "device_id": "device_id",
        "epochs": "epoch",
        "learning_rate": "lr",
        "batch_frame_size": "batch_frame_size",
    },
    "wandb": {
        "project": "wandb_project",
        "entity": "wandb_entity",
        "run_name": "wandb_run_name",
        "mode": "wandb_mode",
    },
}


def load_config_defaults(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a YAML mapping")

    defaults = {}
    unknown_sections = sorted(set(raw) - set(CONFIG_SCHEMA))
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {', '.join(unknown_sections)}")
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"config section {section!r} must be a mapping")
        schema = CONFIG_SCHEMA[section]
        unknown_keys = sorted(set(values) - set(schema))
        if unknown_keys:
            raise ValueError(f"unknown key(s) in {section}: {', '.join(unknown_keys)}")
        for key, value in values.items():
            defaults[schema[key]] = value
    return defaults


def build_parser(defaults=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, fine-tune, and evaluate Allosaurus")
    parser.add_argument("--config", type=Path, help="YAML experiment configuration")
    parser.add_argument("--stage", choices=("prepare", "features", "train", "evaluate", "all"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--pretrained-model", default="latest")
    parser.add_argument("--new-model")
    parser.add_argument("--lang", default="kor")
    parser.add_argument("--phone-init-map", default="none")
    parser.add_argument("--phone-mapping", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validate-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-frame-size", type=int, default=6000)
    parser.add_argument("--expected-validate-size", type=int, default=5000)
    parser.add_argument("--wandb-project", default="allosaurus-finetune")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--overwrite", action="store_true")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def prepare_features(manifest_dir: Path, model: str, lang: str, overwrite: bool = False) -> None:
    resolved_model = resolve_model_name(model)
    if resolved_model == "none":
        raise ValueError(f"model does not exist: {model}")
    for split in ("train", "validate"):
        split_dir = manifest_dir / split
        if not (split_dir / "wave").exists() or not (split_dir / "text").exists():
            raise ValueError(f"missing wave/text manifest in {split_dir}")
        if overwrite:
            for name in GENERATED_FEATURE_FILES:
                path = split_dir / name
                if path.exists():
                    path.unlink()
        existing_features = [name for name in ACOUSTIC_FEATURE_FILES if (split_dir / name).exists()]
        if existing_features and len(existing_features) != len(ACOUSTIC_FEATURE_FILES):
            raise FileExistsError(
                f"partial acoustic features in {split_dir}: {', '.join(existing_features)}; "
                "remove them or pass --overwrite"
            )
        if not existing_features:
            prepare_feature(split_dir, resolved_model)
        if not (split_dir / "token").exists():
            prepare_token(split_dir, resolved_model, lang)


def run_training(
    manifest_dir: Path,
    pretrained_model: str,
    new_model: str,
    lang: str,
    device_id: int,
    epoch: int,
    learning_rate: float,
    batch_frame_size: int,
    expected_validate_size: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_run_name: str,
    wandb_mode: str,
    phone_init_map: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "allosaurus.bin.adapt_model",
        f"--pretrained_model={pretrained_model}",
        f"--new_model={new_model}",
        f"--path={manifest_dir}",
        f"--lang={lang}",
        f"--device_id={device_id}",
        f"--epoch={epoch}",
        f"--lr={learning_rate}",
        f"--batch_frame_size={batch_frame_size}",
        f"--expected_validate_size={expected_validate_size}",
        f"--wandb_project={wandb_project}",
        f"--wandb_entity={wandb_entity}",
        f"--wandb_run_name={wandb_run_name}",
        f"--wandb_mode={wandb_mode}",
        f"--phone_init_map={phone_init_map}",
    ]
    subprocess.run(command, check=True)


def main() -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    try:
        defaults = load_config_defaults(config_args.config) if config_args.config else {}
    except (OSError, ValueError, RuntimeError) as exc:
        config_parser.error(str(exc))
    parser = build_parser(defaults)
    args = parser.parse_args()

    if args.stage is None:
        parser.error("--stage is required (or set pipeline.stage in the config)")
    if args.work_dir is None:
        parser.error("--work-dir is required (or set data.work_dir in the config)")
    if args.new_model is None:
        parser.error("--new-model is required (or set model.new_model in the config)")

    manifest_dir = args.work_dir / "manifests"
    evaluation_dir = args.work_dir / "evaluation"
    stages = {args.stage} if args.stage != "all" else {"prepare", "features", "train", "evaluate"}
    try:
        if "prepare" in stages:
            if args.data_root is None:
                parser.error("--data-root is required for prepare/all")
            summary = prepare(
                args.data_root,
                manifest_dir,
                args.pretrained_model,
                args.lang,
                args.phone_mapping,
                args.train_ratio,
                args.validate_ratio,
                args.seed,
                16000,
                10.0,
                args.overwrite,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        if "features" in stages:
            prepare_features(manifest_dir, args.pretrained_model, args.lang, args.overwrite)
        if "train" in stages:
            run_training(
                manifest_dir,
                args.pretrained_model,
                args.new_model,
                args.lang,
                args.device_id,
                args.epoch,
                args.lr,
                args.batch_frame_size,
                args.expected_validate_size,
                args.wandb_project,
                args.wandb_entity,
                args.wandb_run_name or args.new_model,
                args.wandb_mode,
                args.phone_init_map,
            )
        if "evaluate" in stages:
            metrics = evaluate(args.new_model, args.lang, manifest_dir / "test", evaluation_dir, args.device_id)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except (ValueError, FileExistsError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
