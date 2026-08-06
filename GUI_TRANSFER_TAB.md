# Transfer 탭 — 어떻게 동작하나

구현 완료. 이 문서는 "어떻게 붙일까"가 아니라 "왜 이렇게 되어 있나"를 남긴 것이다.
`cosmos_gui.py`의 `transfer_video()`, `stage_control()`, `clip_shape()`를 보면 된다.

## 먼저 알아야 할 것: 업로드 슬롯은 없다

`python cosmos_transfer.py --probe`를 이 빌드에 돌리면 이렇게 나온다:

```
Binary (file) fields: none — this endpoint takes no upload.
```

`input_reference`, `image_reference`, `video_reference` 셋 다 `format: binary`가 아니라
`string`이다. **이건 고장이 아니라 정상이다.** 이 빌드는 control 비디오를 업로드로
받지 않고, `extra_params` 안에 **서버가 직접 열 경로**로 받는다.

vLLM-Omni의 Cosmos3-Nano 레시피 문서 그대로:

> "No special flags are needed to enable transfer controls."
> "Transfer controls are activated per-request via `extra_params`."

즉 **서버 재시작도, 별도 체크포인트(`Cosmos-Transfer2.5-2B`)도 필요 없다.**
Cosmos3-Nano에 control 분기가 이미 들어 있다.

요청 형태:

```
extra_params={"depth": {"control_path": "/서버가/읽을/경로.mp4",
                        "control_weight": 1.0},
              "max_frames": 357,
              "resolution": "720",
              "num_video_frames_per_chunk": 93,
              "use_resolution_template": false,
              "use_duration_template": false}
```

힌트 키는 `depth` / `edge` / `seg` / `blur` / `wsm`.

이 문서의 이전 판은 `post_video()`에 업로드 슬롯을 하나 더 열고 binary 필드를 런타임에
찾는 `control_field()`를 두라고 적어놨었다. 이 빌드에서는 binary 필드가 하나도 없어서
그 함수는 항상 `None`을 반환한다 — 그 패치는 폐기됐고, `post_video()`는 손대지 않았다.

## 서버의 생성 마감시간 — 긴 클립은 반드시 걸린다

`/v1/videos/sync`는 기본 **600초**가 지나면 504로 끊는다. 93프레임은 통과하지만
357프레임(17초 에피소드)은 넘긴다. 클라이언트의 `TIMEOUT`(3600s)과는 무관하다.

값은 컨테이너 안의 이 한 줄이 정한다
(`vllm_omni/entrypoints/openai/api_server.py:2880`):

```python
VIDEO_SYNC_TIMEOUT_S = float(os.environ.get("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", 600.0))
```

**`VLLM_OMNI_VIDEO_SYNC_TIMEOUT` — `OMNI`가 들어간다.** 검색하면 `VLLM_VIDEO_SYNC_TIMEOUT`
이라고 나오는 글이 여럿 있는데 그 이름으로는 아무 일도 일어나지 않는다(값은 컨테이너
env에 얌전히 앉아 있고 서버는 계속 600초에 끊는다). 소스가 유일한 근거다.

서버를 이 변수와 함께 띄운다:

```
docker run -d --name cosmos-server ... \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  ... vllm serve nvidia/Cosmos3-Nano --omni ...
```

환경변수라 값을 바꾸려면 컨테이너를 다시 만들어야 한다. 모델 캐시가 볼륨에 있으면
재다운로드는 없다.

배치로 수십 편을 돌릴 거라면 비동기 경로(`POST /v1/videos` → `GET /v1/videos/{id}`
폴링 → `/content`)가 구조적으로 맞다. 동기 요청을 10분 넘게 붙잡고 있으면 네트워크가
한 번 끊길 때마다 그 생성분이 날아간다. 아직 구현하지 않았다.

## 경로 브릿지

서버가 파일을 **직접 열기** 때문에, GUI가 받은 업로드는 서버가 볼 수 있는 곳으로
건너가야 한다. 서버가 컨테이너 안에 있으면 그 경로는 **컨테이너 내부 경로**다.
`stage_control()`이 세 가지 방법을 지원한다:

| 환경변수 | 동작 | 재시작 |
|---|---|---|
| `COSMOS_CONTROL_DIR=/host/dir:/container/dir` | 호스트 쪽에 복사하고 컨테이너 경로를 넘긴다. **권장** | 마운트가 없으면 1회 |
| `COSMOS_DOCKER_CONTAINER=<name\|id>` | `docker cp`로 컨테이너 `/tmp`에 넣는다 | 불필요 |
| (둘 다 없음) | 호스트 절대경로를 그대로 넘긴다 | 서버가 컨테이너 밖일 때만 맞음 |

`docker -v`와 같은 `host:container` 문법이고, 콜론이 없으면 양쪽이 같은 경로라는 뜻이다.
현재 설정이 어느 쪽인지는 Transfer 탭 상단에 그대로 표시된다 — 이게 틀리면 실패가
서버 깊은 곳의 경로 에러로 나타나서 원인이 이 화면에 안 보이기 때문이다.

마운트 방식으로 갈 때 서버 실행 명령에 볼륨을 하나 추가한다:

```
docker run ... -v /data/jinju/cosmos_ctrl:/cosmos_ctrl ... vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --host 0.0.0.0 --port 8000 --init-timeout 1800
```

그리고 GUI 쪽에 `export COSMOS_CONTROL_DIR=/data/jinju/cosmos_ctrl:/cosmos_ctrl`.

경로 읽기에서 400/422가 나면 그때만 `--allowed-local-media-path /`를 붙인다.
레시피 문서상으로는 불필요하다.

## 프레임 수와 fps를 왜 못 만지게 해놨나

출력이 정책 학습셋으로 돌아가기 때문에, 생성된 영상의 프레임이 원래 에피소드의 액션
라벨과 1:1로 대응해야 한다. 그래서:

- `num_frames`는 control 클립을 실제로 세어서(`clip_shape()`, 헤더를 믿지 않는다) 그
  값으로 못박는다. UI에 슬라이더가 없다.
- `size`는 control 클립 크기를 16의 배수로 깎아서 쓴다.
- `use_*_template`을 끈다. 서버가 자기 해상도/길이 템플릿으로 덮어쓰면 프레임 수가
  바뀌어 정렬이 깨진다.
- **fps는 원본 에피소드와 같아야 한다.** 다른 탭은 24를 쓰지만 우리 LeRobot 데이터는
  60Hz 수집 + stride 3 = **20**이다. 기본값을 20으로 뒀다.

그래도 돌아온 프레임 수가 다르면 상태창에 `do not train on this clip` 경고를 띄운다.
조용히 넘어가면 증강 데이터가 오염된 채로 학습에 들어간다.

## extra_params가 실제로 받는 필드

컨테이너의 `vllm_omni/diffusion/models/cosmos3/transfer.py`가 유일한 근거다.
**이 dataclass에 없는 키는 예외 없이 조용히 버려진다.**

```python
@dataclass
class Cosmos3TransferConfig:
    hints: dict[str, Cosmos3TransferHint]
    guidance_scale: float | None = None
    control_guidance: float = 1.0
    control_guidance_interval: tuple[float, float] | None = None
    flow_shift: float | None = None
    num_video_frames_per_chunk: int = 93
    num_conditional_frames: int = 1
    max_frames: int = 5000
    show_control_condition: bool = False
    show_input: bool = False
    num_first_chunk_conditional_frames: int = 0
    share_vision_temporal_positions: bool = True
    num_frames: int | None = None
    fps: float | None = None

@dataclass
class Cosmos3TransferHint:
    key: str
    control_path: str | None = None
    control: Any | None = None
    preset_edge_threshold: str = "medium"
    preset_blur_strength: str = "medium"
```

힌트 종류별 권장값(`TRANSFER_DEFAULTS`):

| hint | guidance_scale | control_guidance | flow_shift |
|---|---|---|---|
| edge / blur / depth | 3.0 | 1.5 | 10.0 |
| seg | 3.0 | 2.0 | 10.0 |
| wsm | 1.0 | 3.0 | 10.0 (num_frames 101, fps 10, chunk 101) |

**`control_weight`는 존재하지 않는다.** 이전 판의 이 탭은 그 이름을 힌트 안에 넣고
있었고, 아무 일도 하지 않았다 — 그 값만 1.8과 1.0으로 바꾼 두 실행이 **md5까지 같은**
파일을 냈다(각각 1087초/1084초를 실제로 태우고서). 세기를 조절하는 건
**`control_guidance`이고 최상위 필드**다.

그리고 `guidance_scale`은 다른 탭에서 물려받은 6.0을 쓰고 있었는데, transfer의
권장값은 **3.0**이다.

### 측정된 결함 두 개가 이 기본값들로 설명된다

- `num_conditional_frames = 1` — 다음 chunk가 이전 chunk를 **1프레임만** 보고 시작한다.
  chunk 경계에서 밝기가 계단식으로 갈아타는 원인. 357프레임/chunk 93에서 프레임 185의
  밝기 변화가 평소의 9.8배였고, chunk를 121로 바꾸자 그 계단이 245로 **따라 움직였다**.
- `num_first_chunk_conditional_frames = 0` — 첫 chunk는 참조가 아예 없다. 프레임 1~9의
  밝기 요동(최대 33.6배)이 여기서 온다. 학습에 쓸 때 앞 10프레임은 잘라내는 게 안전하다.

### 여러 힌트를 동시에 줄 수 있다

`ordered_hints`가 `TRANSFER_HINT_KEYS` 순서로 여러 힌트를 돌려주므로 depth + seg 병용이
가능하다(`next(iter(hints))`는 기본값을 어느 힌트에서 가져올지 고르는 데만 쓴다).
깊이 대비가 약한 물체는 seg를 함께 주는 게 답일 수 있다.

## control 비디오는 어디서 오나

`pickplace` repo(`~/pickplace`)에서:

```bash
# 1) 수집 때 저장된 시뮬레이터 ground-truth depth를 PNG 폴더로 빼낸다
python export_depth_control.py --src /data/jinju/bc_data_v12 \
                               --out /data/jinju/depth_ctrl_v12

# 2) 에피소드 하나를 control mp4로 (fps는 60/stride)
python make_control.py --mode depth --fps 20 \
    --depth-dir /data/jinju/depth_ctrl_v12/episode_0000 \
    --out /data/jinju/cosmos_ctrl/episode_0000.mp4
```

`export_depth_control.py`가 `subsample_dataset.py`와 같은 프레임 선택 규칙을 쓰므로
1067 → 357프레임이 되어 학습셋 에피소드와 길이가 맞는다.

depth가 없는 예전 데이터(v11)로 시험만 해보려면 `--mode edge`가 RGB 하나로 된다.

## 참고

- vLLM-Omni Cosmos3-Nano 레시피: `recipes/cosmos3/Cosmos3-Nano.md`
- Seeing Realism from Simulation, arXiv 2605.02757 — depth control, 960x720,
  35 steps, 93 frames/chunk, A800 80GB에서 에피소드당 ~560초. 우리 카드는 96GB라
  720p·189프레임 피크가 ~46GiB로 여유가 있다.
