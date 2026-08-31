# Running Pose Feature Prototype

고정된 측면 러닝 영상을 입력받아 사람의 자세를 추정하고, 달리기 동작을 수치화한 뒤 코칭 리포트로 정리하는 실험용 프로젝트입니다.

이 프로젝트에서 확인하려는 핵심은 세 가지입니다.

1. 2D 영상만으로 반복되는 착지 구간을 찾을 수 있는가
2. Halpe-26 키포인트로 러닝 자세와 좌우 비대칭을 계산할 수 있는가
3. 계산 결과를 근거와 한계가 드러나는 리포트로 바꿀 수 있는가

## Prototype 2의 범위

현재 버전은 **러닝 영상에서 어떤 피처를 계산할 수 있는지 확인하는 기능 검증용 프로토타입**입니다. 영상 입력부터 자세 추정, 피처 계산, 결과 저장, 코칭 리포트 생성까지 한 번에 실행됩니다.

피처 이름만 나열하는 코드는 아닙니다. Halpe-26 좌표에서 16개 피처의 실제 수치를 계산하고, 각 결과에 단위, 참고 범위와의 관계, 사용 표본 수, 계산 방법을 함께 기록합니다. 다만 착지와 toe-off를 2D 좌표 변화로 추정하므로 실험실 장비나 정답 데이터와 정확도를 검증한 완성형 분석기는 아닙니다.

더 발전된 알고리즘과 제품화 코드는 Prototype 3에서 별도로 관리합니다. Prototype 2는 현재 계산 흐름과 한계를 이해하고 다음 구현을 비교하기 위한 기준 버전으로 둡니다.

## 처리 흐름

```text
MP4 영상
  → 프레임 추출
  → RTMDet 사람 검출
  → RTMPose-M Halpe-26 자세 추정
  → 주 대상 추적 및 스켈레톤 렌더링
  → 착지·관절각·리듬·비대칭 계산
  → 러닝 코칭 리포트 생성
```

사람이 검출되지 않거나 화면 경계에 걸친 프레임은 자세 분석 구간에서 제외될 수 있습니다. 따라서 `pose_predictions.json`의 프레임 수는 원본 영상의 전체 프레임 수보다 적을 수 있습니다. 렌더링 영상은 원본 길이를 유지하고, 자세 데이터가 있는 프레임에만 스켈레톤을 표시합니다.

## 실행 환경

- Python 3.12 이상
- `uv`
- `ffmpeg`
- ONNX Runtime
- CPU, CUDA 또는 MPS

프로젝트의 Python 환경은 `PoC/.venv`를 사용합니다.

```bash
cd ~/Running_Projects/PoC
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

`ffmpeg`는 운영체제의 패키지 관리자로 별도 설치해야 합니다.

## 입력 준비

사용자가 직접 촬영한 측면 러닝 영상을 준비합니다. 분석 이름과 같은 폴더를 `PoC/run` 아래에 만들고 MP4 파일을 넣습니다.

```text
PoC/run/test1/
└── test1.mp4
```

영상 파일명은 자유롭지만 실행 이름은 폴더명과 같아야 합니다.
촬영 영상과 개인 측정값은 개인정보 보호를 위해 Git에 포함하지 않습니다.

촬영할 때는 다음 조건을 권장합니다.

- 카메라를 주행 방향과 수직인 측면에 고정
- 영상 전체에서 머리부터 양발까지 보이도록 촬영
- 한 명의 러너와 여러 보폭이 포함되도록 촬영
- 발이 가려지지 않고 관절을 구분할 수 있는 밝기 유지
- 줌, 패닝, 카메라 이동 없이 촬영

프레임레이트가 높을수록 착지 시점과 시간 관련 피처를 더 세밀하게 추정할 수 있지만, 이 프로젝트의 결과는 영상 조건에 따라 달라질 수 있습니다.

## 전체 실행

```bash
cd ~/Running_Projects/PoC
./scripts/main.sh test1 --extract --device cpu
```

각 인자의 의미는 다음과 같습니다.

- `test1`: `run/test1`을 분석 대상으로 사용
- `--extract`: MP4에서 입력 프레임 생성
- `--device cpu`: ONNX 추론 장치 선택 (`cpu`, `cuda`, `mps`)

코칭 단계가 필요하지 않으면 다음과 같이 실행합니다.

```bash
./scripts/main.sh test1 --agent false --extract --device cpu
```

## 단계별 실행

이미 생성된 결과를 재사용하면서 특정 단계만 다시 확인할 수 있습니다.

```bash
# 자세 추정과 렌더링
./scripts/hpe/hpe.sh test1 --extract --device cpu

# 러닝 피처 계산
./scripts/features/features.sh test1

# 코칭 리포트 생성
./scripts/Agent/agent.sh test1
```

진행 방향 자동 판정이 틀린 경우 피처 단계에서 방향을 지정합니다.

```bash
.venv/bin/python scripts/features/feature_extract.py \
  . test1 --running-direction left
```

`--running-direction`은 `auto`, `left`, `right` 중 하나입니다.

## 결과 파일

모든 결과는 실행별 `outputs` 폴더에 생성됩니다.

```text
PoC/run/test1/outputs/
├── details.json             # 영상 FPS, 크기, 전체 프레임 수
├── pose_predictions.json    # 분석 구간의 Halpe-26 좌표와 신뢰도
├── rendered/                # 프레임별 스켈레톤 이미지
├── rendered.mp4             # 스켈레톤을 합성한 영상
├── feature_results.json     # 계산된 러닝 지표와 품질 정보
└── running_report.md        # 코칭 리포트
```

OpenAI API를 사용하려면 실행 전에 키를 환경 변수로 지정합니다.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

키가 없거나 API 호출에 실패하면 동일한 피처 결과를 사용해 로컬 규칙형 리포트를 생성합니다. API 키는 `.env` 또는 환경 변수로만 관리하며 Git에는 추가하지 않습니다.

## 계산하는 러닝 피처

`feature_results.json`에는 총 16개 피처가 저장됩니다.

자세:

- 전신 전방 기울기
- 몸통 전방 굴곡

착지와 입각기:

- 착지 시 무릎 굴곡
- 입각기 최대 무릎 굴곡
- 착지 후 무릎 굴곡 변화량
- 착지 시 정강이와 발의 기울기
- 후기 입각기 고관절 신전

리듬과 좌우 차이:

- 케이던스
- 스텝 시간 비대칭
- 추정 입각 시간 비대칭
- 무릎 가동범위 비대칭
- 오버스트라이드 비대칭

탐색용 피처:

- 골반-발목 수평 거리의 신체 크기 비율
- 뒤꿈치-COM 근사 수평 거리의 신체 크기 비율
- 팔꿈치 각도 가동범위

거리 관련 값은 카메라 보정 없이 실제 길이로 변환하지 않고 신체 크기로 정규화한 비율로만 제공합니다. 탐색용 피처는 비교 가능한 수치만 제공하며 정상·비정상 판정에는 사용하지 않습니다.

각 피처 결과는 다음 정보를 가집니다.

| 필드 | 의미 |
|---|---|
| `value` | 계산된 대표값 |
| `unit` | 각도, 비율, steps/min 등의 단위 |
| `status` | 참고 범위 안·밖, 탐색용 또는 데이터 부족 상태 |
| `reference` | 비교에 사용한 연구 참고 범위와 해석 주의사항 |
| `sample_count` | 대표값 계산에 사용한 착지 또는 구간 수 |
| `method` | 2D 좌표 측정, 보폭 간격 추정, 신체 크기 정규화 등의 계산 방식 |

결과의 `status`는 연구 표본과 비교하기 위한 표시일 뿐, 러닝 능력이나 부상을 진단하는 판정값이 아닙니다.

## 코드 구성

```text
PoC/scripts/
├── main.sh                  # 전체 단계 실행
├── hpe/
│   ├── extract_frames.py    # 영상 분해와 메타데이터 저장
│   ├── hpe_model.py         # 사람 검출과 Halpe-26 추론
│   ├── pose_track.py        # 주 대상 선택과 누락 키포인트 표시
│   ├── render.py            # 키포인트를 원본 프레임에 합성
│   └── compose_video.py     # 렌더링 프레임을 MP4로 변환
├── features/
│   ├── feature_extract.py   # 착지 탐지와 러닝 지표 계산
│   └── papers.py            # 지표별 연구 참고 범위
└── Agent/
    ├── Running_coach.py     # OpenAI 또는 로컬 리포트 선택
    ├── local_report.py      # API 없이 생성하는 규칙형 리포트
    └── prompts.py           # OpenAI 리포트 지침
```

## 테스트

```bash
cd ~/Running_Projects/PoC
.venv/bin/python -m unittest discover -s tests -v
```

테스트는 각도 계산, 착지 후보 탐색, 합성 러닝 데이터의 피처 스키마, 짧거나 빈 입력 처리, 로컬 리포트 생성을 확인합니다.

## 분석 조건과 한계

- 카메라는 움직이지 않고 주행 방향과 수직인 측면에 두는 것을 전제로 합니다.
- 머리부터 양발까지 보이고 여러 보폭이 포함된 영상이 필요합니다.
- Initial Contact와 toe-off는 2D 좌표 변화로 추정한 이벤트입니다.
- 픽셀 좌표만으로 실제 거리나 지면 반력을 측정할 수 없습니다.
- 연구 참고 범위는 비교를 위한 값이며 의료 진단 기준이 아닙니다.
- 통증이나 부상 판단에는 이 결과만 사용하지 않습니다.

## 사용 기술과 출처

- 사람 검출: OpenMMLab RTMDet
- 자세 추정: OpenMMLab RTMPose-M, Halpe-26
- 추론과 시각화: ONNX Runtime, RTMLib, OpenCV
- 선택적 리포트 생성: OpenAI API

모델과 라이브러리를 재배포하거나 공개 저장소로 전환하기 전에는 각 프로젝트와 모델 가중치의 라이선스를 다시 확인해야 합니다.
