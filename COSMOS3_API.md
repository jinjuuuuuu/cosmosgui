# Cosmos3 GUI가 서버에 보내는 것

`cosmos_gui.py`가 vLLM-Omni 서버에 무엇을 요청하는지, 각 값이 무엇을 뜻하는지 정리한 문서다.

서버: `http://localhost:8000` (`COSMOS_SERVER`) · 모델 `nvidia/Cosmos3-Nano`

## 쓰는 엔드포인트

| 경로 | 형식 | 어디서 |
|---|---|---|
| `POST /v1/videos/sync` | multipart/form-data | Video · Transfer 탭 |
| `POST /v1/images/generations` | application/json | Image 탭 |
| `GET /v1/models` | — | 연결 확인 |

History 탭은 서버를 부르지 않는다. 결과 폴더(`COSMOS_OUT_DIR`, 워크스테이션은
`/data/cosmos/outputs`)에 쌓인 파일과 그 옆의 `.json`을 읽는다.

## 파라미터 사전

### 모든 탭 공통

| 필드 | 뜻 |
|---|---|
| `prompt` | 만들 장면. 유일한 필수 항목 |
| `negative_prompt` | 결과에서 피하고 싶은 것. 비어 있으면 아예 보내지 않는다 |
| `size` | `가로x세로`. 16의 배수여야 안전하다 |
| `seed` | 난수 시드. 같은 프롬프트 · 같은 값 → 같은 결과 |
| `num_inference_steps` | 확산 반복 횟수. 높을수록 정교하고 오래 걸린다 |
| `guidance_scale` | 프롬프트를 얼마나 강하게 따를지 (CFG). 높이면 문장대로, 낮추면 자유롭게 |
| `flow_shift` | 플로우 매칭 스케줄러의 시그마 이동값. 영상은 10.0, 이미지는 3.0이 권장값이라 탭마다 다르게 들어가 있다 |

### 영상 전용

| 필드 | 뜻 |
|---|---|
| `num_frames` | 총 프레임 수. **길이(초) = num_frames ÷ fps** |
| `fps` | 초당 프레임 |
| `generate_sound` | 장면에 맞는 소리를 만들어 mp4 안에 함께 넣는다 (`"true"` / `"false"`) |
| `input_reference` | 시작 이미지. multipart 파일로 올린다. 넣으면 그 사진에서 이어지는 영상이 된다 |

### 이미지 전용

| 필드 | 뜻 |
|---|---|
| `n` | 생성 장수. **Cosmos3는 값과 무관하게 한 장만 만든다.** 1로 고정해 보낸다 |

## 탭별 요청 필드

**서버 범위**는 `/openapi.json`에 선언된 최소·최대다. `없음`은 서버가 범위를 정하지 않았다는
뜻으로, 아무 값이나 받는다는 것이지 아무 값이나 잘 동작한다는 뜻은 아니다.

GUI의 각 항목은 여기 적힌 필드 이름 그대로 표시되므로, 화면에서 본 이름으로 이 표를 찾으면
된다. `GUI n ~ m`은 슬라이더가 허용하는 폭이며 우리가 정한 것이라 서버와는 무관하다.

### Video 탭 → `POST /v1/videos/sync`

| 필드 | 서버 범위 | 비고 |
|---|---|---|
| `prompt` | — | 필수 |
| `negative_prompt` | — | 비어 있으면 아예 보내지 않는다 |
| `size` | 없음 | 832x480 · 1280x720 · 1024x1024 · 1920x1080 · Match source image |
| `num_frames` | 없음 | GUI 17 ~ 121 (4 단위) |
| `fps` | 없음 | GUI 8 ~ 30 |
| `num_inference_steps` | 없음 | GUI 10 ~ 60 |
| `guidance_scale` | 없음 | GUI 1.0 ~ 12.0 |
| `flow_shift` | 없음 | GUI 1.0 ~ 14.0 |
| `seed` | 없음 | 자유 입력 |
| `generate_sound` | — | 체크박스. `"true"` / `"false"` 문자열로 보낸다 |
| `input_reference` | — | 올렸을 때만 보낸다 |

이 엔드포인트는 숫자 범위를 하나도 선언하지 않는다. 위 GUI 범위는 전부 우리가 정한 것이다.

`Match source image`를 고르면 올린 사진의 가로세로 비율을 유지한 크기를 계산해서 보낸다
(픽셀 수는 832×480 근처로 맞춰 생성 시간을 비슷하게 유지한다). 사진이 없으면 `832x480`.

기본값 기준 길이는 57 ÷ 24 ≈ **2.4초**, 최대 121 ÷ 24 ≈ **5.0초**.

### Image 탭 → `POST /v1/images/generations`

| 필드 | 서버 범위 | 비고 |
|---|---|---|
| `prompt` | — | 필수 |
| `negative_prompt` | — | 비어 있으면 아예 보내지 않는다 |
| `size` | 없음 | 832x480 · 1280x720 · 1024x1024 · 1920x1080 |
| `n` | **1 ~ 10** | 화면에 없다. **Cosmos3가 이 값을 쓰지 않아** 올려도 한 장이라 `1`로 고정해 보낸다 |
| `num_inference_steps` | **1 ~ 200** | GUI 10 ~ 60 |
| `guidance_scale` | **0 ~ 20** | GUI 1.0 ~ 12.0 |
| `flow_shift` | 없음 | GUI 1.0 ~ 12.0 |
| `seed` | 없음 | 자유 입력 |

이 엔드포인트만 범위를 선언한다. `num_inference_steps`와 `guidance_scale`은 **서버가 허용하는
폭이 GUI보다 넓다** — 슬라이더가 좁게 잡혀 있는 것이지 서버가 막는 게 아니다. 반대로 `n`은
범위가 선언돼 있어도 Cosmos3가 그 값을 쓰지 않아서, 올려도 결과는 한 장이다.

영상 관련 항목(`num_frames`, `fps`, `generate_sound`)이 없다. 그래서 훨씬 빠르다.

### Transfer 탭 → `POST /v1/videos/sync` + `extra_params`

이미 있는 영상의 움직임과 형태는 그대로 두고 겉모습만 바꾼다.

**form 필드**

Video 탭과 같은 엔드포인트이므로 서버가 선언한 범위는 여기서도 없다.

| 필드 | 서버 범위 | 비고 |
|---|---|---|
| `model` | — | `nvidia/Cosmos3-Nano`를 명시해서 보낸다 |
| `prompt` | — | 원하는 겉모습. 움직임과 형태는 그대로 둔다 |
| `negative_prompt` | — | `blurry, distorted, low quality, jittery, deformed, warped geometry`가 기본값으로 채워져 있다 |
| `size` | 없음 | **고를 수 없다.** control 영상 크기를 16의 배수로 깎아 보낸다 |
| `num_frames` | 없음 | **고를 수 없다.** control 영상의 실제 프레임 수를 세어 보낸다 |
| `fps` | 없음 | GUI 8 ~ 30 |
| `num_inference_steps` | 없음 | GUI 10 ~ 60 |
| `guidance_scale` | 없음 | GUI 1.0 ~ 12.0 |
| `flow_shift` | 없음 | GUI 1.0 ~ 14.0 |
| `seed` | 없음 | 자유 입력 |
| `generate_sound` | — | 화면에 없다. `false` 고정 — transfer는 소리를 받지 않는다 |

크기와 프레임 수를 고정하는 이유는, 결과 영상이 원래 에피소드의 동작 기록과 1:1로 맞아야
하기 때문이다. 어느 한쪽이 달라지면 그 짝이 깨진다. `fps`도 원본 에피소드와 같아야 한다 —
우리 LeRobot 데이터는 60Hz 수집에 stride 3이라 **20**이다.

<br>
<br>

**`extra_params`** (JSON 문자열로 직렬화해서 form에 넣는다)

`extra_params`는 스키마상 그냥 문자열이라, 그 안의 키에는 **선언된 범위라는 것이 없다.**

| 키 | 뜻 | 비고 |
|---|---|---|
| `depth` / `edge` / `seg` / `blur` / `wsm` | control 영상의 종류 | 라디오(`extra_params hint key`)로 고른 하나에 `{"control_path": "서버가 읽을 경로"}` |
| `control_guidance` | 원본의 형태를 얼마나 단단히 붙잡을지 | GUI 0.5 ~ 4.0 |
| `num_conditional_frames` | 다음 청크가 이전 청크를 몇 프레임 보고 시작할지 | GUI 1 ~ 16 |
| `num_video_frames_per_chunk` | 한 번에 생성하는 청크 크기 | GUI 17 ~ 121 |
| `max_frames` | 총 길이 상한 | control 영상의 프레임 수를 보낸다 |
| `resolution` | 생성 해상도 기준 | control 영상 높이를 16의 배수로 깎아 보낸다 |
| `use_resolution_template` | 서버가 자기 해상도 템플릿으로 덮어쓰지 못하게 | `false` 고정 |
| `use_duration_template` | 서버가 자기 길이 템플릿으로 덮어쓰지 못하게 | `false` 고정 |

화면의 **extra_params** 칸에 JSON을 넣으면 위 값들 위에 덮어써서 보낸다.

**control 영상을 서버에 전달하는 방법** — 서버가 파일을 직접 열기 때문에 서버가 볼 수 있는
곳에 있어야 한다. 셋 중 하나로 동작하며, 현재 어느 쪽인지는 Transfer 탭 상단에 표시된다.

| 환경변수 | 동작 |
|---|---|
| `COSMOS_CONTROL_DIR=/호스트/경로:/컨테이너/경로` | 호스트 쪽에 복사하고 컨테이너 경로를 넘긴다 (권장) |
| `COSMOS_DOCKER_CONTAINER=<이름>` | `docker cp`로 컨테이너 `/tmp`에 넣는다 |
| (둘 다 없음) | 호스트 절대경로를 그대로 넘긴다 |

## 응답

**영상** — `Accept: video/mp4`로 요청하므로 mp4 바이트가 그대로 온다. 서버가 실제로 계산에
쓴 시간은 `X-Inference-Time-S` 헤더에 담겨 오고, 화면의 "GPU 초"가 이 값이다. 화면에 함께
표시되는 총 시간에는 대기와 파일 전송이 포함된다.

**이미지** — JSON의 `data[0].b64_json`에 base64로 담겨 온다.

**저장** — 결과 파일 옆에 같은 이름의 `.json`을 함께 쓴다. 프롬프트, 크기, 각 설정값, 시드,
소요 시간, 생성 시각이 들어간다. History 탭이 보여 주는 게 이 파일이고, 이 값을 그대로 다시
입력하면 같은 결과를 재현할 수 있다.

## 참고

- 한 번에 한 건만 처리한다. 모든 탭이 같은 GPU 슬롯을 공유하고, 대기열은 20개까지 받는다.
- `/v1/videos/sync`는 서버 기본 설정상 600초가 지나면 끊긴다
  (`VLLM_OMNI_VIDEO_SYNC_TIMEOUT`). 긴 Transfer 클립은 이 값을 늘려서 서버를 띄워야 한다.
- 가드레일은 켜져 있다. 프롬프트 검사와 함께, 생성물에 사람 얼굴이 있으면 모자이크 처리된다.
