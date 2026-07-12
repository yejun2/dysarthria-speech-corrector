# Allosaurus fine-tuning pipeline

The pipeline expects recursively nested, same-name WAV/IPA pairs:

```text
<DATA_ROOT>/KsponSpeech_01/preprocessed/ipa/KsponSpeech_000001.wav
<DATA_ROOT>/KsponSpeech_01/preprocessed/ipa/KsponSpeech_000001.txt
```

Each TXT file must contain one whitespace-separated IPA sequence. The source data
is never modified. Generated manifests, features, logs, and evaluation results are
written below `--work-dir`.

## End-to-end smoke test

Start with a small data root and a new model name:

```bash
python scripts/run_finetune.py \
  --stage all \
  --data-root /path/to/korean_speech_data \
  --work-dir workspace/kspon_smoke \
  --pretrained-model uni2005 \
  --new-model kspon_smoke_v1 \
  --lang kor \
  --device-id 0 \
  --epoch 2 \
  --expected-validate-size 5000 \
  --wandb-project allosaurus-finetune \
  --wandb-run-name kspon-smoke-v1
```

Log in once before an online W&B run with `wandb login`. Use
`--wandb-mode offline` when the training machine has no network access. Each
epoch validates all 5,000 validation utterances; training aborts before model
creation if the manifest has a different validation size.

The same stages can be run independently:

```bash
python scripts/run_finetune.py --stage prepare  --data-root /path/to/data --work-dir workspace/kspon_v1 --new-model kspon_v1
python scripts/run_finetune.py --stage features --work-dir workspace/kspon_v1 --new-model kspon_v1
python scripts/run_finetune.py --stage train    --work-dir workspace/kspon_v1 --new-model kspon_v1 --device-id 0
python scripts/run_finetune.py --stage evaluate --work-dir workspace/kspon_v1 --new-model kspon_v1 --device-id 0
```

## Reproduce on another machine

Generated manifests and acoustic features are intentionally excluded from Git.
After cloning the repository, create the exact-size split from that machine's
dataset path:

```bash
python scripts/create_data_splits.py \
  --data-root /path/on/that/machine/processed_data \
  --output-dir workspace/kspon_v1/manifests \
  --validate-count 5000 \
  --test-count 1000 \
  --seed 42
```

Then generate features and train from the versioned experiment config:

```bash
python scripts/run_finetune.py --config configs/experiments/kspon_v1.yaml --stage features
python scripts/run_finetune.py --config configs/experiments/kspon_v1.yaml --stage train
```

Manifest `wave` files contain machine-specific absolute WAV paths, so copying a
local `workspace/` to a different filesystem is not supported. Regenerate it
with the same seed instead.

`--overwrite` replaces generated manifests or features, but it does not delete an
existing installed model. Use a unique `--new-model` for every training run.

## Unsupported IPA phones

Data preparation stops when an IPA token is absent from the selected language
inventory. An explicit JSON mapping can replace or drop tokens:

```json
{
  "ɡ": "k",
  "unwanted-token": null
}
```

Pass it with `--phone-mapping mappings/kor.json`. Source TXT files remain unchanged.

## Outputs

```text
<WORK_DIR>/
├── manifests/
│   ├── train/{wave,text,feat.ark,feat.scp,shape,token}
│   ├── validate/{wave,text,feat.ark,feat.scp,shape,token}
│   ├── test/{wave,text}
│   └── summary.json
└── evaluation/{metrics.json,predictions.jsonl}
```

Every completed training epoch is saved under
`allosaurus/pretrained/<NEW_MODEL>/checkpoints/`. `model.pt` continues to contain
the checkpoint with the best validation phone error rate.
