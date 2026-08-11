"""
Cosmos 3 web GUI — a thin client for a local vLLM-Omni server.

Start the server first:

  docker run --runtime nvidia --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -e HF_HUB_DISABLE_XET=1 \
    -v /data/hf_cache:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-omni:cosmos3 \
    vllm serve nvidia/Cosmos3-Nano --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --host 0.0.0.0 --port 8000 --init-timeout 1800

Then:

  pip install gradio requests pillow
  export COSMOS_USERS='jjhan:pw1,nhlee:pw2'      # required, no default accounts
  python cosmos_gui.py

Teammates open http://<server-ip>:7860 — VPN is the only thing they need. No
HF token, no docker, no terminal on their side.

Environment:
  COSMOS_USERS    required. "user:pass" pairs, comma separated.
  COSMOS_SERVER   vLLM-Omni endpoint (default http://localhost:8000)
  COSMOS_OUT_DIR  where generated mp4s are written (default: system temp dir)

  For the Transfer tab only — the server opens the control video itself rather
  than receiving an upload, so the file has to be somewhere it can reach:
  COSMOS_CONTROL_DIR        "/host/dir:/container/dir" — a directory mounted
                            into the container. Preferred.
  COSMOS_DOCKER_CONTAINER   container name or id; the control clip is copied in
                            with `docker cp`. No mount and no restart needed,
                            but this process needs docker permission.
  Set neither only if the server shares this filesystem (not containerised).
"""

import base64
import glob
import io
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time

import gradio as gr
import requests
from PIL import Image

SERVER = os.environ.get("COSMOS_SERVER", "http://localhost:8000")
TIMEOUT = 3600  # generation can take a long while; don't let requests give up

IMAGE_PATH = "/v1/images/generations"
VIDEO_PATH = "/v1/videos/sync"
# Generated videos are kept on disk, so they must not pile up in /tmp on a
# server that runs for weeks. On the workstation point this at /data.
OUT_DIR = os.environ.get("COSMOS_OUT_DIR") or tempfile.gettempdir()
os.makedirs(OUT_DIR, exist_ok=True)

# Transfer needs the control clip to be readable by the *server*, not by us.
# See stage_control() for what these do and why there are two of them.
CONTROL_DIR = os.environ.get("COSMOS_CONTROL_DIR", "")
DOCKER_CONTAINER = os.environ.get("COSMOS_DOCKER_CONTAINER", "")


def save_meta(path, meta):
    """Write the settings next to the artifact.

    Without this, a folder of mp4s is unreadable a week later — nobody can tell
    which prompt or seed produced which file, so nothing is reproducible.
    """
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(path):
    try:
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def upload_field(name, path):
    """A (field, (filename, handle, mimetype)) tuple for a multipart upload.

    The caller owns the handle and must close it.
    """
    fh = open(path, "rb")
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return (name, (os.path.basename(path), fh, mime)), fh


def keep_input(out_path, src):
    """Copy the uploaded image next to the result.

    An augmented image is meaningless without the original beside it, and the
    Gradio upload lives in a temp dir that is cleared on restart.
    """
    if not src or not os.path.exists(src):
        return None
    ext = os.path.splitext(src)[1] or ".png"
    dst = os.path.splitext(out_path)[0] + "_input" + ext
    shutil.copyfile(src, dst)
    return dst


def input_of(path):
    """The saved input image for a result, if this run had one."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.splitext(path)[0] + "_input" + ext
        if os.path.exists(p):
            return p
    return None


def recent_choices(limit=40):
    """Newest artifacts first, as (label, path) pairs for a dropdown."""
    files = [
        p for p in glob.glob(os.path.join(OUT_DIR, "cosmos_*.*"))
        if not p.endswith(".json")
        and "_input." not in os.path.basename(p)
        and "_frame" not in os.path.basename(p)
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    choices = []
    for p in files[:limit]:
        prompt = (load_meta(p).get("prompt") or "").replace("\n", " ").strip()
        stamp = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        kind = "video" if p.lower().endswith(".mp4") else "image"
        choices.append((f"{stamp}  [{kind}]  {prompt[:70] or os.path.basename(p)}", p))
    return choices


def show_run(path):
    """Play back one past run: the input, the result, and the settings used."""
    if not path or not os.path.exists(path):
        return None, None, None, ""
    meta = load_meta(path)
    text = json.dumps(meta, ensure_ascii=False, indent=2) if meta else "(no metadata)"
    src = input_of(path)
    if path.lower().endswith(".mp4"):
        return path, None, src, text
    return None, path, src, text


def accounts():
    """Login pairs from the environment.

    No hardcoded defaults on purpose: this server is reachable by everyone on
    the VPN, and this file gets copied between machines and into git.
    """
    raw = os.environ.get("COSMOS_USERS", "").strip()
    if not raw:
        raise SystemExit(
            "COSMOS_USERS is not set. Example:\n"
            "  export COSMOS_USERS='jjhan:somepassword,nhlee:otherpassword'"
        )
    pairs = []
    for item in raw.split(","):
        if ":" not in item:
            raise SystemExit(f"COSMOS_USERS entry is not user:pass -> {item!r}")
        user, password = item.split(":", 1)
        pairs.append((user.strip(), password))
    return pairs


def server_paths():
    """Endpoint paths this server actually exposes, or an empty set if we
    cannot ask. Cosmos3-Nano is a video model and some builds do not serve an
    image endpoint at all, so the Image tab cannot be assumed to work."""
    try:
        r = requests.get(f"{SERVER}/openapi.json", timeout=5)
        r.raise_for_status()
        return set(r.json().get("paths", {}))
    except Exception:
        return set()


def check_server():
    try:
        r = requests.get(f"{SERVER}/v1/models", timeout=5)
        r.raise_for_status()
        names = [m.get("id", "?") for m in r.json().get("data", [])]
    except Exception as e:
        return f"No server at {SERVER}. It is still starting, or it died. ({e})"

    msg = f"Server is up. Loaded: {', '.join(names) or 'unknown'}"
    paths = server_paths()
    if paths and IMAGE_PATH not in paths:
        msg += f"\nThis server does not expose {IMAGE_PATH} — use the Video tab."
    return msg


def explain_error(r, path):
    """Turn a non-200 into something a teammate can act on."""
    body = r.text[:1500]
    if r.status_code == 404:
        return (
            f"This server does not expose {path} (404).\n"
            "Cosmos3-Nano is a video model — try the Video tab."
        )
    if r.status_code == 422:
        return (
            f"The server rejected a parameter (422). Check the field names "
            f"against {SERVER}/docs.\n\n{body}"
        )
    if "guardrail" in body.lower() or "safety" in body.lower():
        return f"Blocked by the safety filter. Try rewording the prompt.\n\n{body}"
    return f"Server returned {r.status_code}:\n{body}"


def server_seconds(r):
    """The server's own inference time, which /v1/videos/sync reports in a
    header. More honest than our round trip, which also counts queue wait and
    the transfer of a multi-megabyte mp4."""
    try:
        return float(r.headers.get("X-Inference-Time-S"))
    except (TypeError, ValueError):
        return None


def make_image(prompt, negative, size, steps, guidance, flow_shift, seed):
    """Text to a single still.

    Two fields this endpoint advertises are deliberately not exposed, because
    both were measured and neither does what the schema implies. See the note
    on "n" below, and:

    use_system_prompt — meant to let the server rewrite a short prompt into a
    fuller one. The named values are not distinguishable: dynamic, en_vanilla,
    en_recaption, en_think_recaption and en_unified all returned the same
    image for one prompt at one seed. Only omitting the field differs, so the
    six choices are really an on/off. And the rewritten text never comes back
    — revised_prompt and cot_output are in the response but stayed null — so
    a result generated this way cannot be explained or reproduced. Not worth a
    control. (The literal string "None" is also rejected with a 400: the
    allowed value there is a JSON null, which the schema prints as None.)
    """
    if not prompt.strip():
        return None, "Enter a prompt first."
    payload = {
        "prompt": prompt,
        "size": size,
        # The schema advertises 1-10, but this build only ever returns one:
        # the cosmos3 pipeline has no num_images_per_prompt, unlike the
        # flux2_klein and glm_image ones in the same container. Asking for
        # more was silently answered with one, so the field stays at 1 and
        # the tab no longer offers a control that does nothing.
        "n": 1,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "flow_shift": float(flow_shift),
        "seed": int(seed),
    }
    if negative.strip():
        payload["negative_prompt"] = negative
    t0 = time.monotonic()
    try:
        r = requests.post(f"{SERVER}{IMAGE_PATH}", json=payload, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach {SERVER}. Is the container running?"

    if r.status_code != 200:
        return None, explain_error(r, IMAGE_PATH)

    try:
        b64 = r.json()["data"][0]["b64_json"]
    except (KeyError, IndexError, ValueError):
        return None, f"Unexpected response shape:\n{r.text[:1500]}"

    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    took = time.monotonic() - t0
    served = server_seconds(r)

    path = os.path.join(OUT_DIR, f"cosmos_img_{int(time.time())}_{int(seed)}.png")
    img.save(path)
    save_meta(path, {
        "kind": "image",
        "prompt": prompt,
        "negative_prompt": negative.strip() or None,
        "size": size,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "flow_shift": float(flow_shift),
        "seed": int(seed),
        "gpu_seconds": round(served, 1) if served else None,
        "round_trip_seconds": round(took, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    timing = f"{served:.0f}s on the GPU ({took:.0f}s total)" if served else f"{took:.0f}s"
    return img, (
        f"Done in {timing}. {img.width}x{img.height}, seed {int(seed)}.\n"
        f"Saved to {path}"
    )


def fit_size(path, fallback="832x480", budget=832 * 480, step=16, cap=1920):
    """A WxH string that keeps the source image's aspect ratio.

    Resolution is otherwise a fixed choice, so a portrait or square upload gets
    stretched into 16:9. The pixel count is held near the fallback so the
    generation takes about as long as it would have.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return fallback
    if not w or not h:
        return fallback
    scale = (budget / float(w * h)) ** 0.5
    fit = lambda v: max(step, min(cap, int(round(v * scale / step)) * step))
    return f"{fit(w)}x{fit(h)}"


def post_video(form, start_image):
    """POST the video endpoint. Returns (mp4 bytes, gpu seconds, error text).

    Shared by the Video tab and the Augment tab — both are the same request,
    they only differ in what they do with the frames that come back.
    """
    # A list of tuples rather than a dict, because an uploaded file and the
    # plain text fields have to travel in the same multipart body.
    fields = [(k, (None, v)) for k, v in form.items()]
    handle = None
    if start_image:
        # input_reference is the binary slot; image_reference takes a path on
        # the server, which would need --allowed-local-media-path to read.
        field, handle = upload_field("input_reference", start_image)
        fields.append(field)
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
        if handle:
            handle.close()

    if r.status_code != 200:
        return None, None, explain_error(r, VIDEO_PATH)
    # An mp4 starts with a box length then the 'ftyp' tag. A JSON error body
    # would fail this, which is the case we actually want to catch.
    if r.content[4:8] != b"ftyp":
        return None, None, f"Expected mp4 bytes, got:\n{r.text[:1500]}"
    return r.content, server_seconds(r), None


def make_video(prompt, negative, size, num_frames, fps, steps, guidance, flow_shift,
               seed, sound, start_image):
    if not prompt.strip():
        return None, "Enter a prompt first."
    if size == MATCH_SOURCE:
        size = fit_size(start_image) if start_image else "832x480"
    # Everything goes as multipart form fields, so values must be strings —
    # including the boolean, which FastAPI parses from "true"/"false".
    form = {
        "prompt": prompt,
        "size": size,
        "num_frames": str(int(num_frames)),
        "fps": str(int(fps)),
        "num_inference_steps": str(int(steps)),
        "guidance_scale": str(float(guidance)),
        "flow_shift": str(float(flow_shift)),
        "seed": str(int(seed)),
        "generate_sound": "true" if sound else "false",
    }
    if negative.strip():
        form["negative_prompt"] = negative

    t0 = time.monotonic()
    content, served, err = post_video(form, start_image)
    if err:
        return None, err

    path = os.path.join(OUT_DIR, f"cosmos_vid_{int(time.time())}_{int(seed)}.mp4")
    with open(path, "wb") as f:
        f.write(content)
    took = time.monotonic() - t0
    kept = keep_input(path, start_image)
    save_meta(path, {
        "kind": "video (image to video)" if start_image else "video",
        "prompt": prompt,
        "negative_prompt": negative.strip() or None,
        "start_image": os.path.basename(kept) if kept else None,
        "size": size,
        "num_frames": int(num_frames),
        "fps": int(fps),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "flow_shift": float(flow_shift),
        "seed": int(seed),
        "generate_sound": bool(sound),
        "gpu_seconds": round(served, 1) if served else None,
        "round_trip_seconds": round(took, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    timing = f"{served:.0f}s on the GPU ({took:.0f}s total)" if served else f"{took:.0f}s"
    mode = "from your image" if start_image else "from text"
    return path, (
        f"Done in {timing} ({mode}). {int(num_frames)} frames at {fps}fps, {size}, "
        f"seed {int(seed)}.\nSaved to {path}"
    )


def augment_image(source, prompt, size, count, steps, guidance, seed):
    """One photo in, several varied photos out.

    Cosmos3-Nano has no image-to-image path, but it does have image-to-video.
    So we generate a short clip starting from the uploaded photo and keep
    evenly spaced frames — each one is the same scene a moment later, which is
    exactly the variation we want for building a training set.
    """
    if not source:
        return None, "Upload an image first."
    if not prompt.strip():
        return None, "Describe the variation you want."
    try:
        import imageio.v3 as iio
    except ImportError:
        return None, ("Frame extraction needs imageio.\n"
                      "  pip install imageio imageio-ffmpeg")

    if size == MATCH_SOURCE:
        size = fit_size(source)
    count = int(count)
    # Ask for more frames than we keep so the samples are spread out, and stay
    # inside the range the Video tab already proved works.
    n_frames = max(17, min(121, count * 3))

    form = {
        "prompt": prompt,
        "size": size,
        "num_frames": str(n_frames),
        "fps": "24",
        "num_inference_steps": str(int(steps)),
        "guidance_scale": str(float(guidance)),
        "flow_shift": "10.0",
        "seed": str(int(seed)),
        "generate_sound": "false",   # nothing here needs audio
    }

    t0 = time.monotonic()
    content, served, err = post_video(form, source)
    if err:
        return None, err

    stem = os.path.join(OUT_DIR, f"cosmos_aug_{int(time.time())}_{int(seed)}")
    mp4 = stem + ".mp4"
    with open(mp4, "wb") as f:
        f.write(content)

    frames = list(iio.imiter(mp4))
    if not frames:
        return None, "The clip came back empty — nothing to extract."
    step = max(1, len(frames) // count)
    picked = frames[::step][:count]

    out = []
    for i, fr in enumerate(picked, 1):
        p = f"{stem}_frame{i:02d}.png"
        Image.fromarray(fr).save(p)
        out.append(p)

    took = time.monotonic() - t0
    kept = keep_input(mp4, source)
    save_meta(mp4, {
        "kind": "image augmentation",
        "prompt": prompt,
        "source_image": os.path.basename(kept) if kept else None,
        "size": size,
        "frames_generated": len(frames),
        "frames_kept": len(out),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "seed": int(seed),
        "gpu_seconds": round(served, 1) if served else None,
        "round_trip_seconds": round(took, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    timing = f"{served:.0f}s on the GPU ({took:.0f}s total)" if served else f"{took:.0f}s"
    return out, (
        f"Done in {timing}. {len(out)} variations from {len(frames)} frames, {size}.\n"
        f"Saved to {stem}_frame01..{len(out):02d}.png"
    )


def clip_shape(path):
    """(frames, width, height) of a clip, or (None, None, None).

    Counted by walking the frames rather than read from the container header:
    this number has to match the action labels exactly, and headers lie often
    enough that trusting one would corrupt a training set silently.
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


def control_dirs():
    """(host dir, container dir) out of COSMOS_CONTROL_DIR.

    Same "host:container" shape as `docker -v`, and as there, no colon means the
    two are the same path. Split on the *last* colon so a Windows drive letter in
    the host path is not mistaken for the separator.
    """
    if ":" not in CONTROL_DIR:
        return CONTROL_DIR, CONTROL_DIR
    host, _, guest = CONTROL_DIR.rpartition(":")
    return host, (guest or host)


def control_bridge_note():
    """How the control clip will reach the server, in one phrase for the UI.

    Worth showing rather than hiding: if this is wrong the generation fails deep
    inside the server with a path error, and the cause is not on this page.
    """
    if CONTROL_DIR:
        host, guest = control_dirs()
        return f"copied into `{host}`, given to the server as `{guest}`"
    if DOCKER_CONTAINER:
        return f"`docker cp` into container `{DOCKER_CONTAINER}`"
    return ("handed over as a plain host path — only correct if the server is not "
            "in a container. Set COSMOS_CONTROL_DIR or COSMOS_DOCKER_CONTAINER")


def stage_control(src):
    """Put the control clip where the *server* can open it.

    Returns (path to hand the server, how it got there, error).

    This endpoint has no binary upload slot at all — every reference field is a
    string, and a control video is named by path inside extra_params. When the
    server runs in a container that path is a container path, so the file has to
    cross over first. Three ways, in the order to prefer them:

      COSMOS_CONTROL_DIR=/host/dir:/container/dir
          A directory already mounted into the container. Nothing to install,
          nothing to restart once the mount exists.
      COSMOS_DOCKER_CONTAINER=<name|id>
          docker cp into the running container. Needs neither a mount nor a
          restart, but this process needs permission to talk to docker.
      neither
          Hand over the host path unchanged. Correct only when the server is
          not containerised, or genuinely shares this filesystem.
    """
    name = f"control_{int(time.time())}_{os.path.basename(src)}"
    if CONTROL_DIR:
        host_dir, guest_dir = control_dirs()
        try:
            os.makedirs(host_dir, exist_ok=True)
            shutil.copyfile(src, os.path.join(host_dir, name))
        except Exception as e:
            # Anything at all here is a misconfigured env var, not a bug worth
            # crashing a Gradio handler over — the browser would show a stack
            # trace instead of the one thing the user needs to change.
            return None, None, (f"Cannot write the control clip into "
                                f"COSMOS_CONTROL_DIR ({host_dir!r}): {e}")
        return f"{guest_dir.rstrip('/')}/{name}", f"{host_dir} -> {guest_dir}", None

    if DOCKER_CONTAINER:
        guest = f"/tmp/{name}"
        try:
            cp = subprocess.run(["docker", "cp", src, f"{DOCKER_CONTAINER}:{guest}"],
                                capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            return None, None, f"Could not run docker cp: {e}"
        if cp.returncode != 0:
            return None, None, (f"docker cp into {DOCKER_CONTAINER} failed:\n"
                                f"{(cp.stderr or '')[:500]}")
        return guest, f"docker cp -> {DOCKER_CONTAINER}:{guest}", None

    return os.path.abspath(src), "host path as-is", None


def transfer_video(control, prompt, negative, control_type, control_guidance, fps,
                   chunk, cond_frames, steps, guidance, flow_shift, seed, extra_json):
    """Control-conditioned augmentation: a depth/edge/seg clip in, an RGB clip out.

    Unlike the Augment tab this keeps the geometry *and* the timing of the source
    episode, so the original action labels stay valid — which is the whole point
    when the output goes back into a policy training set.

    The control clip is not uploaded. It is named by path inside extra_params:
        extra_params={"depth": {"control_path": "/...", "control_weight": 1.0}, ...}
    which is how this build takes transfer hints. See stage_control().
    """
    if not control:
        return None, "Upload a control video (depth / edge / segmentation) first."
    if not prompt.strip():
        return None, "Describe what the scene should look like."

    n, w, h = clip_shape(control)
    if not n:
        return None, ("Could not read that control video. Frame counting needs "
                      "imageio (pip install imageio imageio-ffmpeg), and the clip "
                      "must be a codec it can decode — re-encode as h264 mp4 if "
                      "in doubt (make_control.py writes h264).")
    # The model works in multiples of 16; trimming is safer than padding because
    # padding would move the image content relative to the control geometry.
    size = f"{w - w % 16}x{h - h % 16}"

    server_path, how, err = stage_control(control)
    if err:
        return None, err

    form = {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": prompt,
        "size": size,
        # Locked to the control clip on purpose. Any other value desynchronises
        # the output from the action trajectory it is supposed to pair with.
        "num_frames": str(n),
        # Must match the source episode, not the 24 the other tabs use. Our
        # LeRobot episodes are 60Hz recorded and strided by 3, so 20.
        "fps": str(int(fps)),
        "num_inference_steps": str(int(steps)),
        "guidance_scale": str(float(guidance)),
        "flow_shift": str(float(flow_shift)),
        "seed": str(int(seed)),
        "generate_sound": "false",   # transfer is video-only; sound is rejected
    }
    if negative.strip():
        form["negative_prompt"] = negative

    # Field names come from Cosmos3TransferConfig in the container
    # (vllm_omni/diffusion/models/cosmos3/transfer.py). Anything not on that
    # dataclass is silently dropped — an earlier version of this tab sent
    # "control_weight" inside the hint and it did nothing at all: two runs that
    # differed only in that value came back byte-identical.
    extra = {
        control_type: {"control_path": server_path},
        # The real name for how hard to hold onto the control geometry.
        "control_guidance": float(control_guidance),
        # How many frames of the previous chunk the next one gets to see. The
        # default of 1 is why the brightness steps across a chunk boundary.
        "num_conditional_frames": int(cond_frames),
        "max_frames": n,
        "resolution": str(h - h % 16),
        "num_video_frames_per_chunk": int(chunk),
        # Keep the server from substituting its own resolution/duration, which
        # would change the frame count and break the label alignment above.
        # (Not on the transfer dataclass, so possibly ignored — the server has
        # been returning 1104x832 for a 960x720 request either way.)
        "use_resolution_template": False,
        "use_duration_template": False,
    }
    if extra_json.strip():
        try:
            extra.update(json.loads(extra_json))
        except ValueError as e:
            return None, f"extra_params overrides are not valid JSON: {e}"

    form["extra_params"] = json.dumps(extra)

    t0 = time.monotonic()
    content, served, err = post_video(form, None)
    if err:
        hint = f"\n\nControl was handed over as: {server_path} ({how})."
        low = err.lower()
        if "timed out" in low or "timeout" in low:
            # The server enforces its own generation deadline (default 600s),
            # which is separate from the TIMEOUT this client waits. Pointing at
            # the control path here would send someone chasing the wrong thing.
            hint += (
                f"\nThat is the server's deadline, not this client's — the "
                f"control path was fine. {n} frames at {int(steps)} steps did not "
                f"finish in time. Lower the steps to get under it, or restart the "
                f"server with a bigger VLLM_VIDEO_SYNC_TIMEOUT (default 600s).")
        else:
            hint += (
                "\nIf the server says it cannot read that path, it is looking at "
                "its own filesystem — set COSMOS_CONTROL_DIR or "
                "COSMOS_DOCKER_CONTAINER (see the tab notes).")
        return None, err + hint

    path = os.path.join(OUT_DIR, f"cosmos_xfer_{int(time.time())}_{int(seed)}.mp4")
    with open(path, "wb") as f:
        f.write(content)
    took = time.monotonic() - t0

    out_n, out_w, out_h = clip_shape(path)
    kept = keep_input(path, control)
    save_meta(path, {
        "kind": f"video transfer ({control_type})",
        "prompt": prompt,
        "negative_prompt": negative.strip() or None,
        "control_video": os.path.basename(kept) if kept else None,
        "control_type": control_type,
        "control_guidance": float(control_guidance),
        "num_conditional_frames": int(cond_frames),
        "control_path_on_server": server_path,
        "size": size,
        "num_frames": n,
        "frames_returned": out_n,
        "fps": int(fps),
        "num_video_frames_per_chunk": int(chunk),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "flow_shift": float(flow_shift),
        "seed": int(seed),
        "gpu_seconds": round(served, 1) if served else None,
        "round_trip_seconds": round(took, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    timing = f"{served:.0f}s on the GPU ({took:.0f}s total)" if served else f"{took:.0f}s"
    # A frame count that came back different is the one failure that does real
    # damage downstream, so say it loudly rather than burying it in the json.
    warn = ""
    if out_n and out_n != n:
        warn = (f"\n\n[!] Asked for {n} frames, got {out_n}. The action labels no "
                f"longer line up — do not train on this clip. Try setting "
                f"num_video_frames_per_chunk to {n} if it fits, or report the "
                f"mismatch before augmenting the rest.")
    return path, (
        f"Done in {timing}. {n} frames at {size}, {control_type} control "
        f"(control_guidance {control_guidance}, {cond_frames} conditional "
        f"frame(s) per chunk), seed {int(seed)}.\n"
        f"Control handed over as {server_path} ({how}).\nSaved to {path}{warn}"
    )


SIZES = ["832x480", "1280x720", "1024x1024", "1920x1080"]
MATCH_SOURCE = "Match source image"
SIZES_WITH_MATCH = [MATCH_SOURCE] + SIZES

# A blank prompt box is where people give up. These are deliberately concrete —
# one subject, one motion, one lighting cue — which is what this model wants.
EXAMPLE_PROMPTS = [
    ["A robot arm slowly picks up a blue box from a conveyor belt and places it on a wooden pallet, industrial warehouse lighting."],
    ["A Franka Panda arm reaches down to a red cube on a grey table, closes its gripper, and lifts the cube, soft studio lighting."],
    ["A wheeled delivery robot drives down a warehouse aisle between tall shelves, overhead fluorescent light, camera follows from behind."],
    ["A human hand places a small metal part into a bin while a robot arm waits beside it, close-up, shallow depth of field."],
    ["An automated forklift lifts a stack of cardboard boxes in a loading dock, morning sunlight through the open door."],
]

# Variation instructions for the Augment tab. Each one keeps the scene and
# changes a single thing, which is what makes the extracted frames useful.
EXAMPLE_AUGS = [
    ["The robot arm slowly moves closer to the blue box."],
    ["The camera pans slowly to the right across the same scene."],
    ["The forklift edges forward while everything else stays still."],
    ["The scene stays put while the light shifts across the floor."],
]

# Transfer prompts describe the *look*, not the motion — the motion is already
# fixed by the control video. Each of these changes lighting, surfaces and
# background while leaving the task itself alone, which is what makes the
# generated episode still valid against its original action labels.
#
# All of them say the cube is on the *floor*, because that is what our depth
# actually shows: the Isaac Sim scene is a Franka standing on a ground plane
# with the cube on that same plane. There is no table. Asking for one puts the
# prompt in a fight with the control geometry, and the control wins — you get a
# smeared surface where the model tried to satisfy both.
EXAMPLE_XFERS = [
    ["A Franka Panda arm reaches down to a red cube lying on a polished concrete floor. Warm late-afternoon sunlight from a window on the left, long soft shadows, brushed aluminium robot. Static third-person camera, photorealistic."],
    ["A Franka Panda arm reaches down to a red cube lying on a scuffed grey epoxy factory floor. Overhead fluorescent panels in an industrial workcell, cool white balance, faint floor markings. Static third-person camera, photorealistic."],
    ["A Franka Panda arm reaches down to a red cube lying on a dark rubber laboratory floor. Dim room light with one bright work lamp from the right, deep shadows, mild sensor noise, slightly underexposed. Static third-person camera, photorealistic."],
    ["A Franka Panda arm reaches down to a red cube lying on a pale linoleum floor in a busy research lab. Bright daylight, cardboard boxes and cable spools far in the background. Static third-person camera, photorealistic."],
]

with gr.Blocks(title="Cosmos 3") as app:
    gr.Markdown(
        "## Cosmos 3\n"
        "Text to video, generated on the workstation GPU.\n\n"
        "One job runs at a time — if someone else is generating, yours waits in "
        "line. A video takes minutes, so leave the tab open."
    )

    status = gr.Textbox(label="Connection", lines=2, interactive=False)
    gr.Button("Recheck connection").click(check_server, outputs=status)

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                v_prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="A robot arm slowly picks up a blue box and moves it to the left.",
                )
                gr.Examples(
                    label="Example prompts — click one to fill the box",
                    examples=EXAMPLE_PROMPTS,
                    inputs=[v_prompt],
                )
                v_start = gr.Image(
                    label="Start image (optional) — leave empty for text to video",
                    type="filepath",
                    sources=["upload", "clipboard"],
                    height=200,
                )
                v_negative = gr.Textbox(label="Negative prompt", lines=2)
                v_size = gr.Dropdown(SIZES_WITH_MATCH, value="832x480", label="Resolution")
                v_frames = gr.Slider(17, 121, value=57, step=4, label="Frames")
                v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                v_sound = gr.Checkbox(value=True, label="Generate sound (muxed into the mp4)")
                v_steps = gr.Slider(10, 60, value=35, step=1, label="Sampling steps")
                v_guidance = gr.Slider(1.0, 12.0, value=6.0, step=0.5, label="Guidance scale")
                v_shift = gr.Slider(1.0, 14.0, value=10.0, step=0.5, label="Flow shift (10.0 for video)")
                v_seed = gr.Number(value=0, precision=0, label="Seed")
                v_go = gr.Button("Generate video", variant="primary")
            with gr.Column():
                v_out = gr.Video(label="Result")
                v_log = gr.Textbox(label="Status", lines=6, interactive=False)

        # Shares one concurrency slot with the Image tab: both put a multi-GB
        # model on the same GPU, and two at once is an out-of-memory crash.
        v_event = v_go.click(
            make_video,
            inputs=[v_prompt, v_negative, v_size, v_frames, v_fps, v_steps,
                    v_guidance, v_shift, v_seed, v_sound, v_start],
            outputs=[v_out, v_log],
            concurrency_id="gpu",
        )

    with gr.Tab("Image"):
        gr.Markdown("A single still frame. Much faster than a video — good for trying out a prompt.")
        with gr.Row():
            with gr.Column():
                i_prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="A warehouse robot arm picking up a blue box, industrial lighting.",
                )
                i_negative = gr.Textbox(label="Negative prompt", lines=2)
                # Cosmos3 keeps a separate set of defaults per mode, and the
                # image ones are not the video ones. From the pipeline in the
                # container (COSMOS3_T2I_DEFAULT_*): 1024x1024, 50 steps,
                # guidance 7.0, flow_shift 3.0. Whatever we send wins over
                # those, so sending the video numbers here quietly opted out
                # of the tuning that ships with the model.
                i_size = gr.Dropdown(SIZES, value="1024x1024", label="Resolution")
                i_steps = gr.Slider(10, 60, value=50, step=1, label="Sampling steps")
                i_guidance = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="Guidance scale")
                i_shift = gr.Slider(1.0, 12.0, value=3.0, step=0.5, label="Flow shift (3.0 for images)")
                i_seed = gr.Number(value=0, precision=0, label="Seed")
                i_go = gr.Button("Generate image", variant="primary")
            with gr.Column():
                i_out = gr.Image(label="Result", type="pil", interactive=False)
                i_log = gr.Textbox(label="Status", lines=6, interactive=False)

        i_event = i_go.click(
            make_image,
            inputs=[i_prompt, i_negative, i_size, i_steps, i_guidance, i_shift, i_seed],
            outputs=[i_out, i_log],
            concurrency_id="gpu",
        )

    with gr.Tab("Augment"):
        gr.Markdown(
            "One photo in, several varied photos out — the same scene a moment "
            "later each time. Useful for building up a training set."
        )
        with gr.Row():
            with gr.Column():
                a_src = gr.Image(label="Source photo", type="filepath",
                                 sources=["upload", "clipboard"], height=240)
                a_prompt = gr.Textbox(
                    label="What should change",
                    lines=3,
                    placeholder="The robot arm slowly moves closer to the blue box.",
                )
                gr.Examples(label="Example instructions", examples=EXAMPLE_AUGS, inputs=[a_prompt])
                a_count = gr.Slider(2, 16, value=8, step=1, label="How many photos to keep")
                a_size = gr.Dropdown(SIZES_WITH_MATCH, value=MATCH_SOURCE, label="Resolution")
                a_steps = gr.Slider(10, 60, value=20, step=1, label="Sampling steps")
                a_guidance = gr.Slider(1.0, 12.0, value=6.0, step=0.5, label="Guidance scale")
                a_seed = gr.Number(value=0, precision=0, label="Seed")
                a_go = gr.Button("Generate variations", variant="primary")
            with gr.Column():
                a_out = gr.Gallery(label="Variations", columns=4, height=430)
                a_log = gr.Textbox(label="Status", lines=5, interactive=False)

        a_event = a_go.click(
            augment_image,
            inputs=[a_src, a_prompt, a_size, a_count, a_steps, a_guidance, a_seed],
            outputs=[a_out, a_log],
            concurrency_id="gpu",
        )

    with gr.Tab("Transfer"):
        gr.Markdown(
            "Domain randomisation for an episode you already have. Upload the "
            "**control** video — depth, edges or segmentation, built with "
            "`make_control.py` — and describe the look you want. The motion and "
            "the geometry stay exactly as they were, so the original action "
            "labels remain valid.\n\n"
            "Frames and resolution follow the control clip and are not adjustable "
            "here, on purpose: changing either one desynchronises the result from "
            "the action trajectory it has to pair with.\n\n"
            f"The server opens the control file itself, so it must be reachable "
            f"from the server's own filesystem — currently "
            f"**{control_bridge_note()}**."
        )
        with gr.Row():
            with gr.Column():
                x_ctrl = gr.Video(label="Control video (depth / edge / segmentation)")
                x_type = gr.Radio(["depth", "edge", "seg", "blur", "wsm"],
                                  value="depth", label="What kind of control is it")
                x_prompt = gr.Textbox(
                    label="Target appearance",
                    lines=5,
                    placeholder="A Franka Panda arm reaches down to a red cube "
                                "lying on a polished concrete floor. Warm afternoon "
                                "sunlight from a window on the left, brushed "
                                "aluminium robot. Static third-person camera, "
                                "photorealistic.",
                )
                gr.Examples(label="Variation ideas — one per augmented copy",
                            examples=EXAMPLE_XFERS, inputs=[x_prompt])
                x_negative = gr.Textbox(
                    label="Negative prompt", lines=2,
                    value="blurry, distorted, low quality, jittery, deformed, warped geometry",
                )
                x_cguidance = gr.Slider(
                    0.5, 4.0, value=1.5, step=0.1,
                    label="Control guidance — how hard to hold the geometry (depth default 1.5)")
                x_fps = gr.Slider(8, 30, value=20, step=1,
                                  label="FPS — must match the source episode (ours is 20)")
                x_chunk = gr.Slider(17, 121, value=121, step=4,
                                    label="Frames per chunk (121 measured cleaner than 93 here)")
                x_cond = gr.Slider(
                    1, 16, value=1, step=1,
                    label="Conditional frames per chunk — how much of the previous "
                          "chunk the next one sees (server default 1)",
                    info="The brightness step at a chunk boundary comes from this "
                         "being 1. Raising it is the lever to try.")
                x_steps = gr.Slider(10, 60, value=35, step=1, label="Sampling steps")
                x_guidance = gr.Slider(1.0, 12.0, value=3.0, step=0.5,
                                       label="Guidance scale (transfer default is 3.0, not 6.0)")
                x_shift = gr.Slider(1.0, 14.0, value=10.0, step=0.5, label="Flow shift")
                x_seed = gr.Number(value=0, precision=0, label="Seed")
                x_extra = gr.Textbox(
                    label="extra_params overrides (JSON, optional)",
                    lines=2,
                    placeholder='{"guardrails": false}',
                    info="Merged over what this tab builds. The escape hatch for "
                         "keys this build wants that we do not know about yet.",
                )
                x_go = gr.Button("Generate augmented video", variant="primary")
            with gr.Column():
                x_out = gr.Video(label="Augmented result")
                x_log = gr.Textbox(label="Status", lines=8, interactive=False)

        x_event = x_go.click(
            transfer_video,
            inputs=[x_ctrl, x_prompt, x_negative, x_type, x_cguidance, x_fps,
                    x_chunk, x_cond, x_steps, x_guidance, x_shift, x_seed, x_extra],
            outputs=[x_out, x_log],
            concurrency_id="gpu",
        )

    with gr.Tab("History"):
        gr.Markdown(
            "Everything generated on this server, newest first. Each result has "
            f"its settings saved beside it, so a run can be repeated exactly.\n\n"
            f"Files live in `{OUT_DIR}`."
        )
        h_refresh = gr.Button("Refresh list")
        h_pick = gr.Dropdown(label="Recent runs", choices=[], value=None)
        with gr.Row():
            h_video = gr.Video(label="Video")
            h_image = gr.Image(label="Image", interactive=False)
            h_src = gr.Image(label="Input image (if this run had one)", interactive=False)
        h_meta = gr.Code(label="Settings used", language="json")

        h_refresh.click(lambda: gr.update(choices=recent_choices()), outputs=h_pick)
        h_pick.change(show_run, inputs=h_pick,
                      outputs=[h_video, h_image, h_src, h_meta])

    # A finished generation adds a file, so the history list is stale until it is
    # rebuilt. Chained with .then so it runs after the job, not alongside it.
    refresh_history = lambda: gr.update(choices=recent_choices())
    v_event.then(refresh_history, outputs=h_pick)
    i_event.then(refresh_history, outputs=h_pick)
    a_event.then(refresh_history, outputs=h_pick)
    x_event.then(refresh_history, outputs=h_pick)

    # Checked on every page load, not once at startup — otherwise everyone who
    # opens the page sees whatever was true when this script happened to start.
    app.load(check_server, outputs=status)
    app.load(refresh_history, outputs=h_pick)

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=accounts(),
        # Results live outside the working directory, and Gradio refuses to
        # serve a path it did not create unless it is declared here. Without
        # this, generation succeeds and then the player cannot load the file.
        allowed_paths=[OUT_DIR],
    )
