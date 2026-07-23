# Allosaurus Korean Phone Recognizer

> KsponSpeech로 파인튜닝한 **한국어 조음장애 음성 교정용 음소 인식 파이프라인**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

이 저장소는 범용 다국어 음소 인식기
[Allosaurus](https://github.com/xinjli/allosaurus)를 기반으로, KsponSpeech에서
구축한 한국어 IPA 데이터로 음향 모델을 파인튜닝하는 부산대학교 학부 졸업
프로젝트입니다.

단순한 학습 스크립트 모음이 아니라 **데이터 무결성 검사 → 재현 가능한 데이터 분할
→ 음향 특징 추출 → 모델 학습 및 재개 → PER 평가 → 추론**을 하나의 실험
파이프라인으로 제공합니다. 최종 모델은 한국어의 자음·모음과 음절 종성에 필요한
34개 음소를 인식하도록 설계했습니다.

> [!IMPORTANT]
> 본 프로젝트는 연구 및 프로토타이핑 목적입니다. 출력된 음소열은 의학적 진단이나
> 언어치료사의 임상적 판단을 대체하지 않습니다.

## 핵심 기능

- **한국어 34음소 인식** — 평음·경음·격음, 치경구개음, 종성 불파음 등을 포함한
  전용 IPA inventory
- **재현 가능한 실험** — 고정 seed와 고정 크기 validation/test 분할, YAML 기반 설정
- **안전한 데이터 처리** — 원본을 수정하지 않고 WAV/IPA 쌍, 인코딩, 채널,
  샘플레이트, 길이, 중복 ID를 검사
- **전이학습과 학습 재개** — `uni2005` 사전 학습 모델, 음소별 가중치 초기화,
  checkpoint 기반 복구 지원
- **안정적인 학습 옵션** — AdamW/SGD, 학습률 scheduler, early stopping,
  encoder freeze, gradient clipping, dropout, time masking
- **정량 평가** — 발화별 예측 결과와 전체 Phone Error Rate(PER)를 JSON으로 저장
- **실험 추적** — Weights & Biases online/offline/disabled 모드 지원

## 파이프라인

```text
KsponSpeech PCM + IPA label
            │
            ▼
     16 kHz mono WAV 변환
            │
            ▼
  데이터 감사 및 inventory 검증
            │
            ▼
 train / validate / test 고정 분할
            │
            ▼
   음향 특징 및 token 생성
            │
            ▼
  Allosaurus uni2005 파인튜닝
            │
            ▼
   best checkpoint 선택 및 PER 평가
            │
            ▼
      한국어 음소열 추론
```

## 프로젝트 구조

```text
allosaurus/
├── allosaurus/                         # Allosaurus 모델, 디코더, 학습 코드
├── configs/
│   ├── experiments/                    # 버전별 재현 가능한 실험 설정
│   └── inventories/
│       ├── kspon_korean_ipa.txt         # 한국어 34음소 inventory
│       └── kspon_phone_init.json        # 신규 음소 가중치 초기화 규칙
├── scripts/
│   ├── audit_finetune_data.py           # WAV/IPA 데이터 무결성 검사
│   ├── build_kspon_processed_data.py    # PCM과 IPA label로 학습 데이터 구성
│   ├── create_data_splits.py            # 고정 크기 데이터 분할
│   ├── extend_kspon_train_manifest.py   # validation/test를 보존하며 train 확장
│   ├── prepare_finetune_data.py          # 비율 기반 manifest 생성
│   ├── run_finetune.py                   # 단계별/전체 파이프라인 실행
│   └── evaluate_finetune.py              # PER 평가
├── docs/finetuning.md                   # 파인튜닝 상세 문서
├── test/                                # 단위 및 회귀 테스트
├── environment.yml                     # 재현용 Conda 환경
└── workspace/                           # 로컬 산출물(버전 관리 제외)
```

## 빠른 시작

### 1. 환경 구성

권장 환경은 Linux, Python 3.10, CUDA 지원 GPU입니다. CPU에서도 실행할 수 있지만
특징 추출과 학습 시간이 크게 늘어납니다.

```bash
git clone https://github.com/yejun2/dysarthria-speech-corrector.git
cd dysarthria-speech-corrector

conda env create -f environment.yml
conda activate allosaurus-ft
```

기존 환경을 최신 정의와 동기화하려면 다음 명령을 사용합니다.

```bash
conda env update -f environment.yml --prune
```

`environment.yml`은 PyTorch 2.5.1과 CUDA 12.1을 기준으로 합니다. GPU 드라이버나
CUDA 구성이 다르다면 `pytorch-cuda` 버전을 시스템 환경에 맞게 조정하세요.

### 2. 사전 학습 모델 준비

```bash
python -m allosaurus.bin.download_model -m uni2005
```

### 3. 데이터 검사

학습 데이터는 같은 이름의 WAV/TXT 쌍으로 구성하며 하위 디렉터리를 재귀적으로
탐색합니다.

```text
<DATA_ROOT>/KsponSpeech_01/preprocessed/ipa/KsponSpeech_000001.wav
<DATA_ROOT>/KsponSpeech_01/preprocessed/ipa/KsponSpeech_000001.txt
```

TXT 파일에는 공백으로 구분한 한 줄의 IPA 음소열이 있어야 합니다.

```text
n a n ɯ n h a k k j o e k a n t a
```

먼저 전체 데이터를 검사합니다.

```bash
python scripts/audit_finetune_data.py /path/to/processed_data \
  --json-output workspace/kspon_v1/audit.json
```

빈 라벨, 다중 행 라벨, 한글 또는 원본 전사 기호가 남은 라벨, 대응 파일이 없는
WAV/TXT는 오류입니다. 16 kHz가 아니거나 10초를 초과하는 음성은 경고로 기록됩니다.
감사 과정은 원본 파일을 변경하지 않습니다.

### 4. 고정 데이터 분할

기본 실험은 validation 5,000개, test 1,000개를 고정하고 나머지를 train으로
사용합니다.

```bash
python scripts/create_data_splits.py \
  --data-root /path/to/processed_data \
  --output-dir workspace/kspon_v1/manifests \
  --validate-count 5000 \
  --test-count 1000 \
  --seed 42
```

동일한 데이터와 seed를 사용하면 같은 발화가 같은 split에 배정됩니다. 단, `wave`
manifest에는 절대 경로가 기록되므로 다른 머신에서는 manifest를 복사하지 말고 같은
명령으로 다시 생성해야 합니다.

### 5. 특징 추출, 학습, 평가

```bash
# train/validate 음향 특징과 token 생성
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage features

# 모델 파인튜닝
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage train

# test split의 PER 평가
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage evaluate
```

CPU에서 실행하려면 `--device-id -1`을 추가합니다. 이미 생성된 manifest나 특징을
다시 만들 때만 `--overwrite`를 사용하세요.

## 데이터 준비

### 준비된 WAV/IPA 쌍을 사용하는 경우

위의 [데이터 검사](#3-데이터-검사)부터 시작하면 됩니다. 음성 권장 형식은 다음과
같습니다.

| 항목 | 권장값 |
|---|---|
| 컨테이너 | WAV |
| 채널 | mono |
| 샘플레이트 | 16 kHz |
| 라벨 인코딩 | UTF-8 |
| 라벨 형식 | 한 줄, 공백 구분 IPA |

### KsponSpeech PCM에서 구성하는 경우

원본 PCM과 별도로 생성한 동일 구조의 IPA label이 있다면 다음 명령으로 16-bit,
16 kHz, mono WAV와 TXT 쌍을 만들 수 있습니다.

```bash
python scripts/build_kspon_processed_data.py \
  /path/to/KsponSpeech_01 \
  /path/to/KsponSpeech_02 \
  --label-root /path/to/ipa_labels \
  --output-root /path/to/processed_data \
  --workers 8
```

이 스크립트는 입력 PCM을 signed 16-bit little-endian, 16 kHz, mono로 간주합니다.
원본 데이터의 실제 포맷이 다르면 변환 전에 반드시 메타데이터를 확인하세요.

## 한국어 음소 체계

`configs/inventories/kspon_korean_ipa.txt`에는 다음 34개 음소가 정의되어 있습니다.

| 구분 | 음소 |
|---|---|
| 모음 | `a`, `e`, `i`, `o`, `u`, `ɯ`, `ʌ` |
| 활음 | `j`, `w` |
| 비음 | `m`, `n`, `ŋ` |
| 유음 | `l`, `ɾ` |
| 마찰음 | `s`, `s͈`, `ɕ`, `ɕ͈`, `h` |
| 파열음 | `p`, `pʰ`, `p͈`, `t`, `tʰ`, `t͈`, `k`, `kʰ`, `k͈` |
| 파찰음 | `tɕ`, `tɕʰ`, `tɕ͈` |
| 종성 불파음 | `p̚`, `t̚`, `k̚` |

inventory에 없는 토큰이 발견되면 데이터 준비를 중단합니다. 표기 변형을 치환하거나
제외해야 한다면 JSON mapping을 만들고 `--phone-mapping`으로 전달할 수 있습니다.

```json
{
  "ɡ": "k",
  "unwanted-token": null
}
```

이 mapping 역시 원본 TXT는 수정하지 않고 생성되는 manifest에만 적용됩니다.

## 실험 설정

모든 주요 하이퍼파라미터는 `configs/experiments/*.yaml`에서 관리합니다. CLI 옵션은
설정 파일의 값을 덮어씁니다.

```yaml
data:
  work_dir: workspace/kspon_v1
  seed: 42
  expected_validate_size: 5000

model:
  pretrained_model: uni2005
  new_model: kspon_ko_v1
  inventory: configs/inventories/kspon_korean_ipa.txt
  phone_initialization: configs/inventories/kspon_phone_init.json

training:
  device_id: 0
  epochs: 10
  learning_rate: 0.001
  batch_frame_size: 6000
```

예를 들어 epoch와 learning rate만 변경하려면 다음과 같이 실행합니다.

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage train \
  --epoch 20 \
  --lr 0.0005 \
  --wandb-mode offline
```

`expected_validate_size`와 실제 validation 발화 수가 다르면 모델을 만들기 전에 학습을
중단합니다. 이는 실수로 다른 validation split을 사용해 실험 간 결과를 비교하는 일을
방지합니다.

### 전체 파이프라인 실행

비율 기반 80:10:10 split으로 빠르게 전체 흐름을 시험하려면:

```bash
python scripts/run_finetune.py \
  --config configs/experiments/kspon_v1.yaml \
  --stage all \
  --data-root /path/to/processed_data
```

`--stage all`의 prepare 단계는 기본적으로 비율 분할을 사용합니다. 따라서 validation을
정확히 5,000개로 유지하는 본 실험에서는 고정 분할을 먼저 만든 뒤 `features`,
`train`, `evaluate`를 순서대로 실행하는 방식을 권장합니다.

### Checkpoint에서 학습 재개

새 실험 설정의 `model.initial_checkpoint`에 기존 checkpoint를 지정합니다.

```yaml
model:
  pretrained_model: uni2005
  new_model: kspon_ko_recovery
  inventory: configs/inventories/kspon_korean_ipa.txt
  initial_checkpoint: allosaurus/pretrained/kspon_ko_v1/checkpoints/epoch_0010.pt
```

그 다음 해당 설정으로 평소와 같이 학습합니다.

```bash
python scripts/run_finetune.py \
  --config configs/experiments/my_recovery.yaml \
  --stage train
```

기존 모델 디렉터리는 자동 삭제하지 않습니다. 실험마다 고유한 `new_model` 이름을
사용해 결과가 섞이지 않도록 하세요.

## 실험 추적

온라인 W&B를 사용할 경우 최초 한 번 로그인합니다.

```bash
wandb login
```

| 모드 | 옵션 | 용도 |
|---|---|---|
| Online | `--wandb-mode online` | W&B 서버에 실시간 기록 |
| Offline | `--wandb-mode offline` | 네트워크 없이 로컬 기록 |
| Disabled | `--wandb-mode disabled` | W&B 기록 비활성화 |

개인 또는 조직 계정에서 실행할 때는 설정 파일의 `wandb.entity`를 자신의 entity로
바꾸거나 `--wandb-entity`로 덮어쓰세요.

## 평가

평가는 전체 test reference에 대한 Levenshtein distance로 PER을 계산합니다.

```text
PER = (치환 수 + 삭제 수 + 삽입 수) / 정답 음소 수
```

결과는 다음 위치에 저장됩니다.

```text
workspace/kspon_v1/evaluation/
├── metrics.json          # 전체 발화 수, edit distance, PER
└── predictions.jsonl     # 발화별 정답, 예측, edit distance
```

PER은 낮을수록 좋습니다. 서로 다른 모델을 비교할 때는 반드시 동일한 test manifest,
inventory, 전처리 조건을 사용하세요. 이 저장소는 측정되지 않은 성능 수치를
README에 임의로 기재하지 않으며, 재현한 실험의 `metrics.json`을 최종 결과로
간주합니다.

## 학습 모델로 추론

### CLI

```bash
python -m allosaurus.run \
  --model kspon_ko_v1 \
  --lang configs/inventories/kspon_korean_ipa.txt \
  --device_id 0 \
  -i /path/to/sample.wav
```

### Python

```python
from allosaurus.app import read_recognizer

recognizer = read_recognizer("kspon_ko_v1")
phones = recognizer.recognize(
    "/path/to/sample.wav",
    "configs/inventories/kspon_korean_ipa.txt",
)

print(phones)
```

출력 예시는 공백으로 구분된 음소열입니다.

```text
n a n ɯ n h a k k j o e k a n t a
```

## 산출물

```text
workspace/<EXPERIMENT>/
├── manifests/
│   ├── train/{wave,text,feat.ark,feat.scp,shape,token}
│   ├── validate/{wave,text,feat.ark,feat.scp,shape,token}
│   ├── test/{wave,text}
│   ├── train.jsonl
│   ├── validate.jsonl
│   ├── test.jsonl
│   └── summary.json
└── evaluation/
    ├── metrics.json
    └── predictions.jsonl

allosaurus/pretrained/<NEW_MODEL>/
├── model.pt                         # validation PER 기준 best model
└── checkpoints/                    # epoch별 checkpoint
```

`workspace/`, 학습 모델, W&B 로그는 크기가 크고 머신별 절대 경로를 포함할 수 있어
Git에서 제외됩니다.

## 테스트

```bash
python -m unittest discover -s test
```

또는 pytest가 설치된 환경에서는:

```bash
pytest -q
```

## 문제 해결

<details>
<summary><strong>CUDA out of memory</strong></summary>

`training.batch_frame_size` 또는 `--batch-frame-size`를 줄이세요. 필요하면
`--device-id -1`로 CPU 실행 여부를 먼저 확인할 수 있습니다.

</details>

<details>
<summary><strong>model does not exist: uni2005</strong></summary>

사전 학습 모델이 설치되지 않은 상태입니다.

```bash
python -m allosaurus.bin.download_model -m uni2005
```

</details>

<details>
<summary><strong>validation manifest size 오류</strong></summary>

설정의 `expected_validate_size`와 실제 validation 발화 수가 다릅니다. 공식 실험은
`create_data_splits.py --validate-count 5000`으로 다시 생성하세요. 작은 smoke
test라면 실제 크기에 맞춰 `--expected-validate-size`를 지정합니다.

</details>

<details>
<summary><strong>unsupported phone 오류</strong></summary>

라벨의 IPA 토큰이 `configs/inventories/kspon_korean_ipa.txt`에 없습니다. 오탈자와
유니코드 표기를 확인한 뒤, 의도한 변형이라면 `--phone-mapping`으로 명시적으로
정규화하세요.

</details>

<details>
<summary><strong>다른 머신에서 WAV 파일을 찾지 못함</strong></summary>

manifest의 WAV 경로는 절대 경로입니다. `workspace/`를 복사하지 말고 새 머신의 데이터
경로에서 동일한 seed로 split과 특징을 다시 생성하세요.

</details>

## 한계와 책임 있는 사용

- KsponSpeech는 일반 한국어 음성 중심이므로 조음장애 음성에 대한 실제 일반화 성능은
  별도의 임상 데이터로 검증해야 합니다.
- PER은 음소열 편집 오류를 측정하지만 발음의 심각도, 명료도, 치료 효과를 직접
  의미하지 않습니다.
- 표준 목표 발음과 화자의 실제 발음 차이, 음운 변동, 장단·억양·음절 경계는 별도의
  발음 사전이나 forced alignment 계층이 필요할 수 있습니다.
- 음성 데이터에는 개인정보가 포함될 수 있습니다. 데이터셋의 이용 조건과 연구
  윤리, 보관 및 비식별화 정책을 준수하세요.

## 참고 자료 및 인용

- [파인튜닝 상세 문서](docs/finetuning.md)
- [원본 Allosaurus 저장소](https://github.com/xinjli/allosaurus)
- [Universal Phone Recognition with a Multilingual Allophone System](https://arxiv.org/abs/2002.11800)
- [KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition](https://aihub.or.kr/)

Allosaurus를 연구에 활용했다면 원 논문을 함께 인용해 주세요.

```bibtex
@inproceedings{li2020allosaurus,
  title     = {Universal Phone Recognition with a Multilingual Allophone System},
  author    = {Li, Xinjian and Dalmia, Siddharth and Li, Juncheng and Lee, Matthew
               and Littell, Patrick and Yao, Jiali and Anastasopoulos, Antonios
               and Mortensen, David R. and Neubig, Graham and Black, Alan W.},
  booktitle = {ICASSP},
  year      = {2020}
}
```

## License

이 프로젝트는 [GNU General Public License v3.0](LICENSE)에 따라 배포됩니다.
기반 프로젝트인 Allosaurus의 저작권과 라이선스 조건도 함께 존중해 주세요.
