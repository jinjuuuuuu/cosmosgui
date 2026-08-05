#!/usr/bin/env python3
"""
Cosmos3-Nano video transfer — control-conditioned video augmentation from the
command line, against the same vLLM-Omni server cosmos_gui.py already talks to.

Two things this does that the GUI cannot:

  1. --probe   Asks the server what fields /v1/videos/sync actually accepts and
               reports which of them look like transfer-control slots. The
               transfer field names are NOT documented in the public recipe, so
               do this once before anything else.

  2. Sends a transfer request: a control video (depth / edge / segmentation)
     plus a prompt, and gets an RGB video back with the geometry preserved.
     Batches over a list of prompt variants so one episode yields N augmented
     versions in a single command.

Quick start
-----------
  # 0) server must be up with local-path reading allowed if you use --control-in extra:
  #    ...  vllm serve nvidia/Cosmos3-Nano --omni --allowed-local-media-path /  ...

  # 1) find the real field names
  python cosmos_transfer.py --probe

  # 2) make a control video from your Isaac Sim clip (see make_control.py)
  python make_control.py --mode edge --rgb episode_000.mp4 --out ctrl_edge.mp4

  # 3) one augmented video
  python cosmos_transfer.py \
      --control ctrl_edge.mp4 --control-type edge \
      --prompt "A Franka Panda arm picks up a red cube from a stainless steel
                table. Warm afternoon sunlight from a window on the left,
                concrete warehouse floor, brushed aluminium robot." \
      --out-dir ./augmented

  # 4) N variants of the same episode
  python cosmos_transfer.py --control ctrl_edge.mp4 --control-type edge \
      --variants variants.json --out-dir ./augmented

Environment
-----------
  COSMOS_SERVER   vLLM-Omni endpoint (default http://localhost:8000)
"""

import argparse
import json
import mimetypes
import os
import re
import sys
import time

import requests

SERVER = os.environ.get("COSMOS_SERVER", "http://localhost:8000").rstrip("/")
VIDEO_PATH = "/v1/videos/sync"
TIMEOUT = 3600

# Substrings that mark a request field as a transfer-control slot. The vLLM-Omni
# pipeline docs name the five hint types; the wrapping field name is what we are
# trying to discover.
CONTROL_HINTS = ("depth", "edge", "seg", "blur", "wsm", "control", "hint")

# Frames the server was tuned for. A control video shorter than this will be
# padded or rejected depending on the build, so we match num_frames to the file.
DEFAULT_NEGATIVE = "blurry, distorted, low quality, jittery, deformed, warped geometry"


# ---------------------------------------------------------------- introspection
#
# All schema reading is delegated to cosmos_schema.py, which resolves $refs and
# reports each field's real type. That matters here: the control slot is
# whichever field is declared `format: binary`, and its NAME may say nothing
# about depth or edges. Matching on the type is right; matching on the name was
# a guess.

def _schema_module():
    """cosmos_schema, if it sits next to this script."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import cosmos_schema
        return cosmos_schema
    except ImportError:
        return None


def video_body():
    """(rows, binary_field_names, error) for POST /v1/videos/sync."""
    cs = _schema_module()
    if cs is None:
        return None, None, ("cosmos_schema.py is not importable — put it in the "
                            "same folder as this script.")
    spec, err = cs.fetch_spec(force=True)
    if not spec:
        return None, None, err
    op = ((spec.get("paths") or {}).get(VIDEO_PATH) or {}).get("post")
    if not isinstance(op, dict):
        return None, None, (f"{VIDEO_PATH} is not served by this build. Run "
                            f"`python cosmos_schema.py` for the endpoint list.")
    _, schema = cs.body_schema(cs.deref(op, spec))
    rows = cs.field_rows(schema)
    binaries = [r[0] for r in rows if "binary" in r[1]]
    return rows, binaries, ""


def resolve_control_field(rows, binaries, requested):
    """The multipart field the control video goes into. Returns (name, note).

    input_reference is the slot the GUI already uses for the I2V start image, so
    any OTHER binary field is the control input. If it is the only binary slot,
    this build has no separate transfer upload and the hint has to travel inside
    extra_params instead.
    """
    names = [r[0] for r in rows] if rows else []
    if requested:
        if names and requested not in names:
            return requested, (f"! '{requested}' is not in the schema. Known "
                               f"binary slots: {', '.join(binaries) or 'none'}")
        return requested, ""

    others = [b for b in binaries or [] if b != "input_reference"]
    if not others:
        return None, ("No binary field other than input_reference on "
                      f"{VIDEO_PATH}. This build takes transfer hints inside "
                      "extra_params, if at all — retry with:\n"
                      "  --control-in extra --control-field depth\n"
                      "and confirm against the container:\n"
                      "  docker exec -it <container> python -c \\\n"
                      "    \"import vllm_omni.model_extras.cosmos3 as m, inspect;"
                      " print(inspect.getsource(m))\" | grep -iE 'depth|edge|seg|control'")
    if len(others) == 1:
        return others[0], f"control field: {others[0]} (only non-image binary slot)"
    ranked = [b for b in others if any(h in b.lower() for h in CONTROL_HINTS)]
    pick = (ranked or others)[0]
    return pick, (f"{len(others)} candidate slots ({', '.join(others)}); using "
                  f"{pick}. Override with --control-field.")


def probe():
    """Report the endpoint's fields, then say what it means for transfer."""
    cs = _schema_module()
    if cs is not None:
        cs.main(["POST", VIDEO_PATH])
        print()
    rows, binaries, err = video_body()
    if err:
        print(err, file=sys.stderr)
        return 1
    print("=== transfer verdict " + "=" * 40)
    print(f"binary slots: {', '.join(binaries) or 'none'}")
    field, note = resolve_control_field(rows, binaries, None)
    if note:
        print(note)
    if field:
        print(f"\nRun with:  --control-in form --control-field {field}")
    return 0


# ------------------------------------------------------------------ generation

def probe_video(path):
    """(frame_count, width, height, fps) of a local mp4, best effort.

    Counts frames by walking them rather than trusting container metadata:
    the header frame count is wrong often enough, and this number has to match
    the action labels exactly.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        return None, None, None, None
    n, w, h = 0, None, None
    try:
        for fr in iio.imiter(path):
            if n == 0:
                h, w = fr.shape[0], fr.shape[1]
            n += 1
    except Exception:
        return None, None, None, None
    fps = None
    try:
        fps = float(iio.immeta(path).get("fps"))
    except Exception:
        pass
    return (n or None), w, h, fps


def build_form(args, num_frames, size):
    form = {
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "size": size,
        "num_frames": str(num_frames),
        "fps": str(args.fps),
        "num_inference_steps": str(args.steps),
        "guidance_scale": str(args.guidance),
        "flow_shift": str(args.flow_shift),
        "max_sequence_length": "4096",
        "seed": str(args.seed),
        "generate_sound": "false",   # transfer is video-only; sound is rejected
    }
    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
    }
    if args.no_guardrails:
        extra["guardrails"] = False

    if args.control_in == "extra":
        # Server-side path reference. Needs --allowed-local-media-path on the
        # server, and the path must be valid *inside* the container.
        key = args.control_field or args.control_type
        extra[key] = {"control_path": args.control_path_in_server or
                      os.path.abspath(args.control),
                      "control_weight": args.control_weight}
    if args.extra_json:
        extra.update(json.loads(args.extra_json))
    form["extra_params"] = json.dumps(extra)
    return form


def send(args, form):
    """POST the transfer request. Returns (mp4 bytes, gpu_seconds, error)."""
    fields = [(k, (None, v)) for k, v in form.items()]
    handle = None
    if args.control_in == "form":
        field = args.control_field
        mime = mimetypes.guess_type(args.control)[0] or "video/mp4"
        handle = open(args.control, "rb")
        fields.append((field, (os.path.basename(args.control), handle, mime)))
    try:
        r = requests.post(f"{SERVER}{VIDEO_PATH}",
                          headers={"Accept": "video/mp4"},
                          files=fields, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        return None, None, f"Cannot reach {SERVER} ({e})"
    finally:
        if handle:
            handle.close()

    if r.status_code != 200:
        hint = ""
        if r.status_code == 422:
            hint = ("\nThe server rejected a field. Run --probe and pass the "
                    "right --control-field / --control-in.")
        return None, None, f"HTTP {r.status_code}{hint}\n{r.text[:1200]}"
    if r.content[4:8] != b"ftyp":
        return None, None, f"Expected mp4 bytes, got:\n{r.text[:1200]}"
    try:
        served = float(r.headers.get("X-Inference-Time-S"))
    except (TypeError, ValueError):
        served = None
    return r.content, served, None


def run_one(args, tag):
    n, w, h, _ = probe_video(args.control)
    num_frames = args.num_frames or n
    if not num_frames:
        sys.exit("Could not read the control video. Pass --num-frames explicitly.")
    size = args.size or (f"{w}x{h}" if w else "1280x720")
    if n and args.num_frames and args.num_frames != n:
        print(f"  ! control video has {n} frames but --num-frames is "
              f"{args.num_frames}; action labels will not line up")

    form = build_form(args, num_frames, size)
    t0 = time.monotonic()
    content, served, err = send(args, form)
    if err:
        print(f"  FAILED: {err}")
        return None

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{tag}_seed{args.seed}")
    mp4 = stem + ".mp4"
    with open(mp4, "wb") as f:
        f.write(content)
    meta = dict(form)
    meta.update({
        "control_video": os.path.abspath(args.control),
        "control_type": args.control_type,
        "control_in": args.control_in,
        "control_field": args.control_field,
        "gpu_seconds": round(served, 1) if served else None,
        "round_trip_seconds": round(time.monotonic() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    took = served or (time.monotonic() - t0)
    print(f"  ok  {took:.0f}s  {num_frames}f {size}  -> {mp4}")
    return mp4


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true",
                   help="print the server's accepted fields and exit")
    p.add_argument("--control", help="control video (depth/edge/seg mp4)")
    p.add_argument("--control-type", default="depth",
                   choices=["depth", "edge", "seg", "blur", "wsm"])
    p.add_argument("--control-in", default="form", choices=["form", "extra"],
                   help="send the control as a multipart upload or as a path "
                        "inside extra_params (default: form)")
    p.add_argument("--control-field",
                   help="exact field name for the control (see --probe)")
    p.add_argument("--control-path-in-server",
                   help="path of the control video as the container sees it, "
                        "for --control-in extra")
    p.add_argument("--control-weight", type=float, default=1.0)

    p.add_argument("--prompt", help="what the augmented scene should look like")
    p.add_argument("--variants", help="JSON file: list of {name, prompt, seed?}")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)

    p.add_argument("--size", help="WxH; default matches the control video")
    p.add_argument("--num-frames", type=int,
                   help="default matches the control video (keep it that way)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--flow-shift", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="nvidia/Cosmos3-Nano")
    p.add_argument("--no-guardrails", action="store_true",
                   help="set guardrails:false per request (only works if the "
                        "server was started WITH guardrails)")
    p.add_argument("--extra-json", help="extra_params overrides as a JSON object")
    p.add_argument("--out-dir", default="./augmented")
    args = p.parse_args()

    if args.probe:
        return probe()
    if not args.control:
        p.error("--control is required (or use --probe)")
    if not os.path.exists(args.control):
        p.error(f"no such control video: {args.control}")
    if not args.prompt and not args.variants:
        p.error("pass --prompt or --variants")

    base = os.path.splitext(os.path.basename(args.control))[0]

    # Resolve the field name from the live schema before generating anything. A
    # batch of six variants must not discover a wrong field name six times over,
    # each after a minute of GPU work.
    if args.control_in == "form":
        rows, binaries, err = video_body()
        if err:
            print(err, file=sys.stderr)
            return 1
        args.control_field, note = resolve_control_field(rows, binaries,
                                                         args.control_field)
        if note:
            print(note)
        if not args.control_field:
            return 1

    if args.variants:
        with open(args.variants, encoding="utf-8") as f:
            variants = json.load(f)
        print(f"{len(variants)} variants from {args.variants}")
        made = []
        for i, v in enumerate(variants, 1):
            args.prompt = v["prompt"]
            args.seed = int(v.get("seed", 1000 + i))
            name = v.get("name", f"var{i:02d}")
            print(f"[{i}/{len(variants)}] {name}")
            out = run_one(args, f"{base}_{name}")
            if out:
                made.append(out)
        print(f"\n{len(made)}/{len(variants)} succeeded in {args.out_dir}")
        return 0 if made else 1

    return 0 if run_one(args, base) else 1


if __name__ == "__main__":
    sys.exit(main())
