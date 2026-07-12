# 한국어 조음장애 음성 교정용 음소 인식기

부산대학교 학부 졸업 프로젝트로, 범용 음소 인식기
[Allosaurus](https://github.com/xinjli/allosaurus)를 KsponSpeech 기반 한국어 IPA
데이터로 파인튜닝합니다. 34개 한국어 음소를 인식하는 모델을 만들기 위해 데이터
검증, 재현 가능한 데이터 분할, 특징 추출, 학습, 평가 과정을 하나의 파이프라인으로
구성했습니다.

## 프로젝트 개요

파이프라인은 다음 순서로 동작합니다.

```text
WAV/IPA 쌍
   └─ 데이터 검증 및 train/validate/test 분할
      └─ 음향 특징과 음소 토큰 생성
         └─ uni2005 모델 파인튜닝
            └─ 테스트 세트 PER(Phone Error Rate) 평가
```

주요 설정은 `configs/experiments/kspon_v1.yaml`에서 관리합니다.

- 사전 학습 모델: `uni2005`
- 파인튜닝 모델명: `kspon_ko_v1`
- 한국어 IPA inventory: `configs/inventories/kspon_korean_ipa.txt` (34개 음소)
- 검증 세트 크기: 5,000개
- 실험 기록: Weights & Biases

## 디렉터리 구조

```text
allosaurus/
├── allosaurus/                         # 음소 인식기 및 학습 코드
├── configs/
│   ├── experiments/kspon_v1.yaml       # KsponSpeech 학습 설정
│   └── inventories/
│       ├── kspon_korean_ipa.txt         # 34개 한국어 IPA 목록
│       └── kspon_phone_init.json        # 신규 음소 가중치 초기화 매핑
├── scripts/
│   ├── audit_finetune_data.py           # WAV/IPA 데이터 검사
│   ├── create_data_splits.py            # 고정 크기 데이터 분할
│   ├── prepare_finetune_data.py          # 비율 기반 manifest 생성
│   ├── run_finetune.py                   # 전체 파이프라인 실행
│   └── evaluate_finetune.py              # PER 평가
├── docs/finetuning.md                   # 파인튜닝 세부 설명
├── test/                                # 파이프라인 단위 테스트
└── workspace/                           # 로컬 manifest, 특징, 평가 결과
```

`workspace/`, 학습 모델, W&B 로그는 용량 및 머신별 경로 문제로 Git에서 제외됩니다.

## 실행 환경

- Python 3.10 권장
- Linux
- CUDA 지원 GPU 권장 (CPU 실행도 가능)
- 입력 음성: mono WAV, 16 kHz 권장

저장소 루트에서 가상환경을 만들고 패키지를 설치합니다.

```bash
conda env create -f environment.yml
conda activate allosaurus-ft
```

이미 환경이 존재한다면 `conda env update -f environment.yml --prune`으로
동기화할 수 있습니다. CUDA 12.1을 사용할 수 없는 환경에서는
`environment.yml`의 `pytorch-cuda` 항목을 시스템에 맞게 변경해야 합니다.

사전 학습 모델을 내려받습니다.

```bash
python -m allosaurus.bin.download_model -m uni2005
```

온라인 W&B 기록을 사용할 경우 한 번 로그인합니다.

```bash
wandb login
```

네트워크 없이 학습하려면 실행 시 `--wandb-mode offline`, 기록을 끄려면
`--wandb-mode disabled`를 사용합니다.

## 데이터 형식

데이터 루트 아래에 같은 이름의 WAV와 TXT가 쌍으로 있어야 합니다. 하위 디렉터리는
재귀적으로 검색합니다.

```text
<DATA_ROOT>/KsponSpeech_01/preprocessed/ipa/KsponSpeech_000001.wav
<DATA_ROOT>/KsponSpeech_01/preprocessed/ipa/KsponSpeech_000001.txt
```

TXT 파일은 한 줄의 공백 구분 IPA 시퀀스여야 합니다.

```text
n a n ɯ n h a k k j o e k a n t a
```

한글, 빈 라벨, 여러 줄 라벨, KsponSpeech 원본 주석 기호가 남은 라벨, 대응 파일이
없는 WAV/TXT는 오류로 처리됩니다. 파이프라인은 원본 데이터를 수정하지 않습니다.

### 데이터 사전 검사

```bash
python scripts/audit_finetune_data.py /path/to/processed_data \
  --json-output workspace/kspon_v1/audit.json
```

검사 항목에는 WAV/TXT 쌍, UTF-8 IPA 라벨, mono 채널, 샘플레이트, 오디오 길이,
중복 ID가 포함됩니다. 샘플레이트가 16 kHz가 아니거나 길이가 10초를 넘는 경우에는
경고를 출력합니다.

## 학습 방법

모든 명령은 저장소 루트에서 실행합니다.

### 1. 데이터 분할 생성

현재 실험은 검증 5,000개, 테스트 1,000개를 고정하고 나머지를 학습 데이터로
사용합니다.

```bash
python scripts/create_data_splits.py \
  --data-root /path/to/processed_data \
  --output-dir workspace/kspon_v1/manifests \
  --validate-count 5000 \
  --test-count 1000 \
  --seed 42
```

같은 데이터와 seed를 사용하면 다른 머신에서도 동일한 분할을 재현할 수 있습니다.
다만 `wave` manifest에는 절대 경로가 저장되므로 머신을 옮긴 뒤에는 다시 생성해야
합니다. 기존 결과를 교체하려면 `--overwrite`를 추가합니다.

### 2. 특징 추출

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage features
```

### 3. 모델 학습

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage train
```

설정 파일의 기본값은 GPU 0, 10 epochs, learning rate 0.001입니다. CLI 인자로
설정값을 덮어쓸 수 있습니다.

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage train \
  --device-id 0 \
  --epoch 20 \
  --lr 0.0005 \
  --wandb-mode offline
```

CPU로 실행하려면 `--device-id -1`을 지정합니다.

### 4. 평가

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage evaluate
```

평가 결과는 다음 위치에 생성됩니다.

```text
workspace/kspon_v1/evaluation/
├── metrics.json          # 전체 PER 및 오류 통계
└── predictions.jsonl     # 발화별 정답과 예측 결과
```

### 전체 파이프라인 한 번에 실행

설정 파일의 `data.data_root`가 `null`이므로 실제 데이터 경로를 CLI로 전달해야 합니다.

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage all \
  --data-root /path/to/processed_data
```

`--stage all`의 `prepare` 단계는 기본적으로 80:10:10 비율 분할을 사용합니다. 검증
세트를 정확히 5,000개로 유지해야 하는 본 실험에서는 위의 고정 분할 명령을 먼저
실행한 뒤 `features`, `train`, `evaluate` 단계를 각각 실행하는 방식을 권장합니다.

## 생성 결과

```text
workspace/kspon_v1/
├── manifests/
│   ├── train/{wave,text,feat.ark,feat.scp,shape,token}
│   ├── validate/{wave,text,feat.ark,feat.scp,shape,token}
│   ├── test/{wave,text}
│   ├── train.jsonl
│   ├── validate.jsonl
│   ├── test.jsonl
│   └── summary.json
└── evaluation/{metrics.json,predictions.jsonl}
```

학습된 모델과 epoch별 체크포인트는 아래에 저장됩니다.

```text
allosaurus/pretrained/kspon_ko_v1/
├── model.pt
└── checkpoints/
```

`model.pt`는 검증 PER가 가장 낮은 체크포인트입니다. 같은 이름의 설치된 모델은
`--overwrite`로 삭제되지 않으므로 새 실험에는 고유한 `--new-model` 이름을
사용합니다.

## 학습 모델 추론

```bash
python -m allosaurus.run \
  --model kspon_ko_v1 \
  --lang configs/inventories/kspon_korean_ipa.txt \
  --device_id 0 \
  -i /path/to/sample.wav
```

Python에서도 사용할 수 있습니다.

```python
from allosaurus.app import read_recognizer

recognizer = read_recognizer("kspon_ko_v1")
phones = recognizer.recognize(
    "/path/to/sample.wav",
    "configs/inventories/kspon_korean_ipa.txt",
)
print(phones)
```

## 테스트

```bash
python -m unittest discover -s test
```

## 참고

- 파인튜닝 상세 문서: [docs/finetuning.md](docs/finetuning.md)
- 원본 Allosaurus 논문: [Universal Phone Recognition with a Multilingual Allophone System](https://arxiv.org/abs/2002.11800)
- 원본 Allosaurus 저장소: [xinjli/allosaurus](https://github.com/xinjli/allosaurus)

이 저장소는 Allosaurus를 기반으로 하며, 원본 라이선스 조건을 따릅니다.
