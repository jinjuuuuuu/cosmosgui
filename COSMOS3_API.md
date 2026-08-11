# Cosmos3-Nano가 실제로 쓰는 API

이 서버가 무엇을 받아들이는지가 아니라, **Cosmos3-Nano가 그중 무엇을 실제로 읽는지**를 적은 문서다.
둘은 다르고, 그 차이 때문에 이미 세 번 헛일을 했다.

측정 시점: 2026-08-11 · 이미지 `vllm/vllm-omni:cosmos3` · 모델 `nvidia/Cosmos3-Nano`

## 왜 스키마를 그대로 믿으면 안 되나

`/openapi.json`은 이 서버가 **서빙할 수 있는 모든 모델**을 위한 한 벌짜리 요청 모델에서 생성된다.
컨테이너 안에는 `cosmos3` 말고도 `flux2_klein`, `glm_image`, `longcat_image`, `qwen_image`,
`wan2_2`, `omnigen2` 등이 들어 있다.

```
스키마 = 이 서버가 받는 필드의 합집합
현실   = 그중 Cosmos3-Nano가 구현한 부분집합
```

거기에 두 가지가 더 겹친다.

- **허용값이 스키마에 안 실린다.** 검사가 `Literal[...]`이 아니라 pydantic validator 함수로
  되어 있으면 `enum`이 생성되지 않는다. `use_system_prompt`가 그랬다.
- **스키마가 틀리기도 한다.** `input_reference`는 `string`에 "binary 없음"이라고 나오지만
  실제로는 multipart 파일 업로드가 된다. Start image가 그걸로 동작한다.

그래서 **스키마는 필드 이름을 찾는 데만 쓰고, 동작은 재서 확인한다.**

| 알고 싶은 것 | 근거 |
|---|---|
| 필드 이름 | `python cosmos_schema.py POST /v1/videos/sync` |
| 허용값 | 일부러 틀린 값을 보내고 **400/422 본문을 읽는다** |
| 실제 동작 | **같은 seed로 A/B, 결과 md5 비교** |

세 번째가 이 저장소의 표준 검증 절차다. `control_weight`, `n`, `enable_frame_interpolation`
셋 다 이 방법으로 걸러냈다.

## 엔드포인트

서버는 42개를 연다. 우리와 관계있는 것만.

### 쓰고 있는 것

| 경로 | 어디서 |
|---|---|
| `POST /v1/videos/sync` | Video · Augment · Transfer 탭 |
| `POST /v1/images/generations` | Image 탭 |
| `GET /v1/models` | 연결 확인 (`check_server`) |
| `GET /openapi.json` | 엔드포인트 존재 확인 (`server_paths`) |

### 쓸 수 있는데 안 쓰는 것

**비동기 영상** — `POST /v1/videos` → `GET /v1/videos/{id}` 폴링 → `GET /v1/videos/{id}/content`,
그리고 `DELETE /v1/videos/{id}`. 요청 필드는 `/v1/videos/sync`와 같다.

`/v1/videos/sync`는 기본 600초에서 504로 끊긴다(`VLLM_OMNI_VIDEO_SYNC_TIMEOUT`).
357프레임 Transfer는 1200초가 걸리므로 동기 경로로는 구조적으로 안 된다. 비동기가 정공법이고,
덤으로 `DELETE`가 있어서 결과 삭제도 서버가 지원한다. 자세한 건 `GUI_TRANSFER_TAB.md` 참고.

**`POST /v1/omni/sleep` · `POST /v1/omni/wakeup`** — 컨테이너를 죽이지 않고 GPU를 비운다.
Isaac Sim이나 학습을 돌려야 할 때 쓸 수 있다.

### 쓸 수 없는 것

**`POST /v1/images/edits`** — 이미지 편집 엔드포인트지만 Cosmos3-Nano로는 못 쓴다.
`image` 필드가 `array[string]`이라 서버가 리스트로 감싸 넘기는데, Cosmos3의 전처리기는
한 장만 받는다.

```
HTTP 500
Cosmos3 preprocessing expected PIL image, numpy array, torch tensor, or path,
got <class 'list'>
```

`reference_image`(단수)만 보내면 `422 Field 'image' or 'url' is required`가 난다.
`url`도 배열이라 같은 문제다. 엔드포인트 쪽 배선이라 클라이언트에서 우회할 방법이 없다.

**따라서 단일 이미지를 넣는 길은 `/v1/videos/sync`의 `input_reference` 하나뿐이다.**
Augment 탭이 "짧은 영상을 만들어 프레임을 골라내는" 방식인 것은 우회가 아니라 유일한 경로다.

나머지(`/v1/chat/completions`, `/v1/audio/*`, `/tokenize`, `/v1/responses` 등)는 이 서버가
다른 모델·모달리티를 위해 여는 것이라 무관하다.

## `/v1/videos/sync` 필드 판정

`multipart/form-data`. 값은 전부 문자열로 보낸다(불리언은 `"true"`/`"false"`).

### 보내고 있는 것 — 전부 동작 확인됨

`prompt` · `negative_prompt` · `size` · `num_frames` · `fps` ·
`num_inference_steps` · `guidance_scale` · `flow_shift` · `seed` ·
`generate_sound` · `input_reference` · `extra_params` · `model`

### 무시되는 것 — 측정으로 확인

| 필드 | 무슨 일이 일어나나 |
|---|---|
| `enable_frame_interpolation` | 켜고 끈 결과가 **md5까지 동일**. `wan2_2` 파이프라인만 구현한다 |
| `frame_interpolation_exp` · `_scale` · `_model_path` | 위와 같음 |

프레임 보간은 `serving_video.py`가 값을 `gen_params`에 얹기까지는 하므로 API 층에 있는 것처럼
보인다. 하지만 그 값을 읽는 건 `wan2_2`뿐이다. **5초 한계를 우회하는 카드가 아니다.**

### 아직 안 재본 것

`seconds` · `lora` · `guidance_scale_2` · `true_cfg_scale` · `boundary_ratio` ·
`audio_reference` · `sound_duration` · `image_reference` · `video_reference` ·
`height` · `width` · `user`

`image_reference`는 서버 파일시스템의 경로를 받는다(읽으려면 `--allowed-local-media-path`가
필요할 수 있다). 업로드 슬롯은 `input_reference` 쪽이다.

## `/v1/images/generations` 필드 판정

이쪽만 `application/json`이다. 그래서 `extra_params`가 문자열이 아니라 객체다 —
multipart에는 중첩 구조를 담을 수 없기 때문이고, `/v1/videos/sync`에서 `json.dumps`를 거치는
이유가 이것이다.

### 무시되는 것 — 측정으로 확인

| 필드 | 무슨 일이 일어나나 |
|---|---|
| `n` | 스키마는 `minimum 1, maximum 10`인데 항상 1장만 온다. `data` 배열에 1개만 담겨 돌아온다 |
| `use_system_prompt` | 값을 구분하지 않는다 |

**`n`** — cosmos3 파이프라인에 `num_images_per_prompt` 배선이 없다. 같은 컨테이너의
`flux2_klein`, `glm_image`에는 있다. `glm_image`는 아예 `num_images_per_prompt=1`로 못박아
두었는데, 확산 모델에서 장수를 늘리면 잠재 텐서가 배수로 커지는 걸 생각하면 납득이 간다.

**`use_system_prompt`** — 짧은 프롬프트를 서버가 늘려 주는 스위치. `dynamic`, `en_vanilla`,
`en_recaption`, `en_think_recaption`, `en_unified` 다섯을 같은 프롬프트·같은 seed로 돌렸더니
결과가 전부 같았다. 다른 건 **필드를 아예 안 보냈을 때뿐**이라, 여섯 개처럼 보이지만 실제로는
켬/끔 두 개다. 게다가 응답의 `revised_prompt`와 `cot_output`이 계속 `null`이라 무엇으로
바뀌었는지 볼 수 없다. **설명할 수 없는 옵션은 UI에 올리지 않는다.**

허용값에 주의. 문자열 `"None"`은 400이 난다.

```
Must be one of: ['dynamic', 'en_vanilla', 'en_recaption',
                 'en_think_recaption', 'en_unified', 'custom', None]
```

마지막 `None`은 문자열이 아니라 **JSON null**이다. 스키마 설명문이 파이썬 `None`을 글자로
찍어 놓은 것이라 문자열로 읽으면 틀린다.

`custom`은 서버를 `--use_system_prompt custom`으로 띄웠을 때만 의미가 있다.

## `extra_params`

`extra_params`에 넣은 dict는 `sampling_params.extra_args`가 되어 파이프라인에 그대로 닿는다
(`pipeline_cosmos3.py`의 `_extra_args`). **요청 모델의 검증을 거치지 않으므로, 스키마에 없는
키를 넣는 유일한 통로이자 오타가 조용히 삼켜지는 자리이기도 하다.**

Cosmos3가 실제로 읽는 키는 아래가 전부다.

| 키 | 쓰임 | 우리가 쓰나 |
|---|---|---|
| `depth` · `edge` · `seg` · `blur` · `wsm` | transfer 힌트 | ✅ Transfer 탭 |
| `control_guidance` | 형태를 얼마나 붙잡을지 | ✅ |
| `num_conditional_frames` | 다음 chunk가 이전 chunk를 몇 프레임 보나 | ✅ |
| `num_video_frames_per_chunk` · `max_frames` | 청크 크기와 총 길이 | ✅ |
| `use_resolution_template` · `use_duration_template` | 서버의 해상도·길이 템플릿 덮어쓰기 방지 | ✅ **읽힌다** |
| `resolution` · `image_size` · `aspect_ratio` | 크기 결정 (`resolution` → `image_size` → 720 순) | 일부 |
| `guardrails` | 요청 단위 가드레일 해제 | ❌ |
| `guidance_interval` | guidance 적용 구간 | ❌ |
| `domain_id` · `domain_name` | 도메인 조건 | ❌ |
| `condition_frame_indexes_vision` · `condition_video_keep` | v2v 조건 프레임 선택 | ❌ |
| `max_sequence_length` | 프롬프트 토큰 상한 | ❌ |
| `audio_duration` · `sound_duration` | 소리 길이 | ❌ |
| `frame_rate` · `resolved_frame_rate` | fps 변형 | ❌ |
| `system_prompt` · `use_system_prompt` | 프롬프트 재작성 (위 참고) | ❌ |
| `action` · `action_mode` · `action_video` · `action_fps` · `action_chunk_size` · `raw_action_dim` · `robot_obs` · `observation` | 로봇 정책 경로 | ❌ |

마지막 줄은 별개 주제다. 이 파이프라인에는 `_build_robolab_policy_inputs`,
`_forward_robolab_policy` 같은 **VLA 정책 경로**가 들어 있고, `extra_params`에 `robot_obs`를
넣으면 켜진다. 아직 안 파봤다.

`use_resolution_template` / `use_duration_template`은 `Cosmos3TransferConfig` dataclass에는
없지만 파이프라인이 `extra`에서 직접 읽는다. Transfer 탭 주석의 "아마 무시될 것"은 틀렸다.

## 가드레일

기본으로 켜져 있다. 입력 쪽은 키워드 blocklist와 Aegis 판정, 출력 쪽은 영상 안전 분류기와
얼굴 블러(RetinaFace, 20×20픽셀 넘는 검출을 모자이크)다.

끄는 방법은 두 층이다(`cosmos3/guardrails.py`의 `is_guardrails_enabled`).

| 층 | 방법 | 비고 |
|---|---|---|
| 서버 | `vllm serve`에 `--no-guardrails` | 가드레일 모델을 아예 로드하지 않는다 |
| 요청 | `extra_params`에 `{"guardrails": false}` | 서버 층이 켜져 있을 때만 유효 |

**서버 층을 끄면 요청 단위로 되살릴 수 없다.** 모델 싱글턴이 채워지지 않기 때문이다.

Transfer 탭 `extra_params` 칸의 placeholder가 `{"guardrails": false}`인 것은 이 공식 사용법을
그대로 적어둔 것이다.

> ⚠️ NVIDIA Open Model License는 안전 가드레일을 우회·비활성화하면 라이선스상의 권리가
> 자동으로 종료된다고 명시한다. 회사 자산으로 돌리는 서버에서는 기술 옵션이 아니라 계약
> 문제다. 우리 용도(로봇 팔·큐브·창고 바닥)에는 걸릴 일이 없으므로 켜 둔다.

## 이미지·영상 입력이 파이프라인에 닿는 경로

`pre_process_func`가 `prompt["multi_modal_data"]`에서 꺼낸다.

```python
raw_image = multi_modal_data.get("image")   # 한 장
raw_video = multi_modal_data.get("video")   # 프레임 리스트
```

- **둘 다 주는 것은 금지**된다(action mode 제외): `Cosmos3 non-action generation accepts
  either image or video input, not both.`
- 크기를 안 주면 입력 이미지의 비율로 정한다. `max_area = 720*1280`, 16의 배수로 내림.
- transfer 여부는 `extra`에 힌트 키가 있는지로 판정한다(`has_transfer_hints`).

## 재현 방법

엔드포인트 목록과 필드 표:

```bash
python cosmos_schema.py                          # 엔드포인트 전체
python cosmos_schema.py POST /v1/videos/sync     # 필드 표 + 요청 스니펫
```

파이프라인이 실제로 읽는 키:

```bash
docker exec cosmos-server grep -oE '(_get_sp_param|_sampling_param|_get_prompt_param|extra\.get|extra_args\.get)\([^)]*"[a-z_0-9]+"' \
  /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py \
  | grep -oE '"[a-z_0-9]+"' | sort -u
```

이 목록은 딕셔너리에서 꺼내는 키만 잡는다. `prompt`, `seed`, `num_inference_steps`처럼 요청
객체의 정식 속성인 것들은 안 걸리므로, 목록에 없다고 안 쓰는 것이 아니다.

어떤 모델이 그 필드를 구현했는지:

```bash
docker exec cosmos-server grep -rn "enable_frame_interpolation" \
  /usr/local/lib/python3.12/dist-packages/vllm_omni/ --include=*.py
```

동작 확인 — 같은 seed로 A/B:

```bash
B="-F prompt=a_robot_arm -F size=832x480 -F num_frames=57 -F fps=24 -F num_inference_steps=10 -F seed=0"
curl -s -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" $B -o /tmp/a.mp4
curl -s -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" $B -F "의심되는필드=값" -o /tmp/b.mp4
md5sum /tmp/a.mp4 /tmp/b.mp4     # 같으면 그 필드는 아무 일도 하지 않는다
```

## 남은 것

- **비동기 영상 경로** — Transfer의 600초 한계를 푸는 유일한 정공법. 미구현.
- **`domain_id` / `domain_name`** — 도메인 조건. 로봇 장면에 맞는 값이 있는지 안 봤다.
- **`guidance_interval`** — transfer의 `control_guidance_interval`과 같은 것인지 미확인.
- **로봇 정책 경로** — `robot_obs`로 켜지는 VLA. 별도 주제.
