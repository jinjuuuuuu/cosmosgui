# cosmos_gui.py에 Transfer 탭 추가하기

세 군데만 고치면 됩니다. 필드 이름은 `cosmos_schema`로 런타임에 찾으므로 하드코딩할 게
없습니다. 다만 시작 전에 `python cosmos_transfer.py --probe`를 한 번 돌려서 이 빌드에
transfer 업로드 슬롯이 있는지 확인하세요 — 없으면 3번 탭을 붙여도 동작하지 않습니다.

---

## 1) `post_video()` — 업로드 슬롯을 두 개로

현재 이 함수는 `input_reference` 파일 하나만 붙일 수 있습니다. transfer는 control 비디오가
추가로 필요하니 슬롯을 하나 더 엽니다. 기존 호출부(`make_video`, `augment_image`)는
인자가 기본값이라 그대로 동작합니다.

필드 이름은 **하드코딩하지 말고** GUI가 이미 쓰고 있는 `cosmos_schema`로 런타임에 찾습니다.
control 슬롯은 이름이 무엇이든 `format: binary`이고, `input_reference`가 아닌 쪽입니다.
컨테이너 이미지가 올라가면서 이름이 바뀌어도 이 방식은 따라갑니다.

```python
import cosmos_schema


def control_field(force=False):
    """The multipart field a transfer control video goes into, or None.

    Resolved from the live schema rather than written down: the name is
    undocumented and moves between builds. input_reference is the I2V start
    image slot the Video tab already uses, so any *other* binary field is the
    control input. If there is no other one, this build has no transfer upload.
    """
    if not force and _CONTROL[0] is not False:
        return _CONTROL[0]
    spec, _ = cosmos_schema.fetch_spec(force=True)
    name = None
    if spec:
        op = ((spec.get("paths") or {}).get(VIDEO_PATH) or {}).get("post")
        if isinstance(op, dict):
            _, schema = cosmos_schema.body_schema(cosmos_schema.deref(op, spec))
            binaries = [r[0] for r in cosmos_schema.field_rows(schema)
                        if "binary" in r[1] and r[0] != "input_reference"]
            # Prefer a name that admits what it is, if the build offers several.
            hinted = [b for b in binaries
                      if any(h in b.lower() for h in
                             ("depth", "edge", "seg", "blur", "wsm", "control"))]
            name = (hinted or binaries or [None])[0]
    _CONTROL[0] = name
    return name


_CONTROL = [False]   # False = not looked up yet; None = looked up, not available


def post_video(form, start_image, control=None):
    """POST the video endpoint. Returns (mp4 bytes, gpu seconds, error text).

    Shared by the Video, Augment and Transfer tabs — the same request, they only
    differ in which reference files ride along and what is done with the result.
    """
    fields = [(k, (None, v)) for k, v in form.items()]
    handles = []
    # input_reference is the binary slot; image_reference takes a path on the
    # server, which would need --allowed-local-media-path to read.
    slots = [("input_reference", start_image)]
    if control:
        field = control_field()
        if not field:
            return None, None, (
                "This server exposes no transfer-control upload slot — the only "
                "binary field is input_reference. Check the API schema tab, or "
                "run: python cosmos_transfer.py --probe")
        slots.append((field, control))
    for name, path in slots:
        if path:
            field, fh = upload_field(name, path)
            fields.append(field)
            handles.append(fh)
    try:
        r = requests.post(
            f"{SERVER}{VIDEO_PATH}",
            headers={"Accept": "video/mp4"},
            files=fields,
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return None, None, f"Cannot reach {SERVER}. Is the container running?"
    finally:
        for fh in handles:
            fh.close()

    if r.status_code != 200:
        return None, None, explain_error(r, VIDEO_PATH)
    if r.content[4:8] != b"ftyp":
        return None, None, f"Expected mp4 bytes, got:\\n{r.text[:1500]}"
    return r.content, server_seconds(r), None
```

`check_server()`가 이미 `server_paths()`를 부르니, 거기에 한 줄 더해서 transfer 가능
여부를 배너에 띄워두면 팀원이 헛수고를 안 합니다:

```python
    if not control_field(force=True):
        msg += "\\nNo transfer-control upload slot on this build — Transfer tab will not work."
```

---

## 2) `transfer_video()` — 새 함수 (`augment_image` 바로 뒤에 붙이기)

```python
def clip_shape(path):
    """(frames, width, height) of an uploaded clip.

    Counted by walking the frames, not read from the container header: this
    number has to match the action labels exactly, and headers lie often enough.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        return None, None, None
    n, w, h = 0, None, None
    try:
        for fr in iio.imiter(path):
            if n == 0:
                h, w = fr.shape[0], fr.shape[1]
            n += 1
    except Exception:
        return None, None, None
    return (n or None), w, h


def transfer_video(control, prompt, negative, control_type, steps, guidance,
                   flow_shift, seed):
    """Control-conditioned augmentation: depth/edge/seg clip in, RGB clip out.

    Unlike the Augment tab this keeps the geometry and the timing of the source
    episode, so the original action labels stay valid — which is the whole point
    when the output is going back into a policy training set.
    """
    if not control:
        return None, "Upload a control video (depth / edge / segmentation) first."
    if not prompt.strip():
        return None, "Describe what the scene should look like."

    n, w, h = clip_shape(control)
    if not n:
        return None, ("Could not read that control video. Re-encode it as h264 "
                      "mp4 (make_control.py does this).")
    size = f"{w - w % 16}x{h - h % 16}"

    form = {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": prompt,
        "size": size,
        # Locked to the control clip on purpose. Any other value desynchronises
        # the output from the action trajectory it is supposed to pair with.
        "num_frames": str(n),
        "fps": "24",
        "num_inference_steps": str(int(steps)),
        "guidance_scale": str(float(guidance)),
        "flow_shift": str(float(flow_shift)),
        "max_sequence_length": "4096",
        "seed": str(int(seed)),
        "generate_sound": "false",   # transfer is video-only; sound is rejected
        "extra_params": json.dumps({
            "use_resolution_template": False,
            "use_duration_template": False,
        }),
    }
    if negative.strip():
        form["negative_prompt"] = negative

    t0 = time.monotonic()
    content, served, err = post_video(form, None, control=control)
    if err:
        return None, err

    path = os.path.join(OUT_DIR, f"cosmos_xfer_{int(time.time())}_{int(seed)}.mp4")
    with open(path, "wb") as f:
        f.write(content)
    took = time.monotonic() - t0
    kept = keep_input(path, control)
    save_meta(path, {
        "kind": f"video transfer ({control_type})",
        "prompt": prompt,
        "negative_prompt": negative.strip() or None,
        "control_video": os.path.basename(kept) if kept else None,
        "control_type": control_type,
        "size": size,
        "num_frames": n,
        "fps": 24,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "flow_shift": float(flow_shift),
        "seed": int(seed),
        "gpu_seconds": round(served, 1) if served else None,
        "round_trip_seconds": round(took, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    timing = f"{served:.0f}s on the GPU ({took:.0f}s total)" if served else f"{took:.0f}s"
    return path, (
        f"Done in {timing}. {n} frames at {size}, {control_type} control, "
        f"seed {int(seed)}.\\nSaved to {path}"
    )
```

`keep_input()`은 확장자 목록이 이미지 전용이라 mp4 control은 `_input.mp4`로 복사되지만
`input_of()`가 못 찾습니다. History 탭에서 control 영상까지 보려면 `input_of()`의 확장자
튜플에 `".mp4"`를 추가하세요.

---

## 3) Transfer 탭 (Augment 탭 뒤에 삽입)

```python
    with gr.Tab("Transfer"):
        gr.Markdown(
            "Domain randomisation for an existing episode. Upload the **control** "
            "video — depth, edges or segmentation, built with `make_control.py` — "
            "and describe the look you want. The motion and the geometry stay "
            "exactly as they were, so the original action labels remain valid.\\n\\n"
            "Frames and resolution follow the control clip; they are not "
            "adjustable here on purpose."
        )
        with gr.Row():
            with gr.Column():
                x_ctrl = gr.Video(label="Control video (depth / edge / segmentation)")
                x_type = gr.Radio(["depth", "edge", "seg"], value="depth",
                                  label="What kind of control is it")
                x_prompt = gr.Textbox(
                    label="Target appearance",
                    lines=5,
                    placeholder="A Franka Panda arm picks up a red cube from a "
                                "stainless steel table. Warm afternoon sunlight "
                                "from a window on the left, concrete floor, "
                                "brushed aluminium robot. Static third-person "
                                "camera, photorealistic.",
                )
                gr.Examples(label="Variation ideas", inputs=[x_prompt], examples=[
                    ["Warm late-afternoon sunlight through a window on the left, long soft shadows, polished concrete floor, stainless steel table, photorealistic."],
                    ["Overhead fluorescent panels in an industrial workcell, cool white balance, scuffed epoxy floor, worn wooden workbench, photorealistic."],
                    ["Dim laboratory light with one bright work lamp from the right, deep shadows, mild sensor noise, slightly underexposed, photorealistic."],
                    ["Busy research lab in bright daylight, cardboard boxes and cable spools against the back wall, scratched white laminate table, photorealistic."],
                ])
                x_negative = gr.Textbox(
                    label="Negative prompt", lines=2,
                    value="blurry, distorted, low quality, jittery, deformed, warped geometry",
                )
                x_steps = gr.Slider(10, 60, value=35, step=1, label="Sampling steps")
                x_guidance = gr.Slider(1.0, 12.0, value=6.0, step=0.5, label="Guidance scale")
                x_shift = gr.Slider(1.0, 14.0, value=10.0, step=0.5, label="Flow shift")
                x_seed = gr.Number(value=0, precision=0, label="Seed")
                x_go = gr.Button("Generate augmented video", variant="primary")
            with gr.Column():
                x_out = gr.Video(label="Augmented result")
                x_log = gr.Textbox(label="Status", lines=6, interactive=False)

        x_event = x_go.click(
            transfer_video,
            inputs=[x_ctrl, x_prompt, x_negative, x_type, x_steps, x_guidance,
                    x_shift, x_seed],
            outputs=[x_out, x_log],
            concurrency_id="gpu",
        )
```

그리고 아래쪽 history 갱신 부분에 한 줄 추가:

```python
    x_event.then(refresh_history, outputs=h_pick)
```

---

## 서버 실행 명령 보완

docstring의 docker 명령에 `--allowed-local-media-path /`가 없습니다. 업로드 방식만 쓸
거면 없어도 되지만, `--control-in extra`(경로 참조) 방식을 쓰거나 나중에 배치 처리를 붙일
때 필요하니 추가해 두는 게 편합니다.

```
  vllm serve nvidia/Cosmos3-Nano --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

96GB 카드에서 720p·189프레임은 피크 ~46GiB이므로 여유가 있습니다. 부족해지면
`--quantization fp8`(~36GB)이나 `--enable-layerwise-offload`.
