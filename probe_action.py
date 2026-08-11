"""Probe Cosmos3-Nano's action modes — does it give actions back, and in what shape?

The pipeline in the container carries an action head and the checkpoint has it
enabled (transformer/config.json: action_gen=true, action_dim=64,
num_embodiment_domains=32). Nothing in the GUI uses it yet. Before building a
tab on top, find out three things:

  1. which request shape actually reaches the action path
  2. what comes back, and where in the response it sits
  3. whether the numbers line up with actions we already recorded

Run it on the workstation, next to the server:

  python probe_action.py --video /data/jinju/episodes/episode_0000.mp4
  python probe_action.py --video ep.mp4 --actions ep_actions.npy
  python probe_action.py --video ep.mp4 --mode policy --frames 57

Why the joint layout is not inferred from the video: the model does not work
out how many joints a Franka has. We say which embodiment it is with
domain_name (or domain_id) and the model answers in that domain's action
format, always padded to 64 slots. raw_action_dim says how many of those slots
are real. So the format is ours to declare; only the values come from the
video. Expect correlation rather than matching numbers — the training data was
not our Isaac Sim setup, camera angle, stride or action convention.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

SERVER = os.environ.get("COSMOS_SERVER", "http://localhost:8000")
TIMEOUT = 3600

# Same two escape hatches the Transfer tab uses: the server opens some files
# itself, so they have to be somewhere it can reach. See cosmos_gui.py.
CONTROL_DIR = os.environ.get("COSMOS_CONTROL_DIR", "")
DOCKER_CONTAINER = os.environ.get("COSMOS_DOCKER_CONTAINER", "")

# From action.py's EMBODIMENT_TO_DOMAIN_ID. Ours is a Franka recorded in
# LeRobot format, so these two are the candidates worth trying.
DOMAINS = ["robomind-franka", "droid_lerobot", "bridge_orig_lerobot", "libero"]


def clip_frames(path):
    """Frame count of a clip, counted rather than read from the header."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    try:
        return sum(1 for _ in iio.imiter(path))
    except Exception:
        return None


def stage_for_server(path):
    """Put the clip where the server can open it. Returns (path, how)."""
    if CONTROL_DIR:
        host, _, container = CONTROL_DIR.partition(":")
        container = container or host
        os.makedirs(host, exist_ok=True)
        dst = os.path.join(host, os.path.basename(path))
        if os.path.abspath(dst) != os.path.abspath(path):
            shutil.copyfile(path, dst)
        return os.path.join(container, os.path.basename(path)), f"mounted dir {CONTROL_DIR}"
    if DOCKER_CONTAINER:
        dst = "/tmp/" + os.path.basename(path)
        subprocess.run(["docker", "cp", path, f"{DOCKER_CONTAINER}:{dst}"], check=True)
        return dst, f"docker cp into {DOCKER_CONTAINER}"
    return os.path.abspath(path), "plain host path (only correct if the server is not containerised)"


def find_number_arrays(obj, path="", out=None, depth=0):
    """Walk a JSON response and report every nested list of numbers.

    The docs say actions come back "through the diffusion output
    payload/metadata envelope" without saying where, so rather than guess a
    key, look for anything numeric and report its shape.
    """
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_number_arrays(v, f"{path}.{k}" if path else k, out, depth + 1)
    elif isinstance(obj, list) and obj:
        shape = []
        probe = obj
        while isinstance(probe, list) and probe:
            shape.append(len(probe))
            probe = probe[0]
        if isinstance(probe, (int, float)) and not isinstance(probe, bool):
            out.append((path, shape))
        else:
            for i, v in enumerate(obj[:2]):
                find_number_arrays(v, f"{path}[{i}]", out, depth + 1)
    return out


def dig(obj, dotted):
    """Fetch a value by the dotted path find_number_arrays reported."""
    for part in dotted.replace("]", "").split("."):
        if "[" in part:
            name, _, idx = part.partition("[")
            if name:
                obj = obj[name]
            obj = obj[int(idx)]
        else:
            obj = obj[part]
    return obj


def attempt(label, form, files=None):
    """One request. Returns (ok, parsed json or text, seconds)."""
    fields = [(k, (None, str(v))) for k, v in form.items()]
    handle = None
    if files:
        field, path = files
        handle = open(path, "rb")
        fields.append((field, (os.path.basename(path), handle, "video/mp4")))
    print(f"\n--- {label}")
    print("    fields:", ", ".join(sorted(form)) + (f", {files[0]}=<upload>" if files else ""))
    t0 = time.monotonic()
    try:
        r = requests.post(
            f"{SERVER}/v1/videos/sync",
            headers={"Accept": "application/json"},   # actions, not mp4 bytes
            files=fields,
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError as e:
        print(f"    cannot reach {SERVER}: {e}")
        return False, None, 0.0
    finally:
        if handle:
            handle.close()
    took = time.monotonic() - t0
    print(f"    HTTP {r.status_code}  {r.headers.get('content-type')}  {took:.0f}s")
    if r.status_code != 200:
        print("   ", r.text[:600].replace("\n", "\n    "))
        return False, None, took
    if r.content[4:8] == b"ftyp":
        print("    got mp4 bytes, not a JSON payload — this path ignored the action mode")
        return False, None, took
    try:
        return True, r.json(), took
    except ValueError:
        print("    body is neither mp4 nor JSON:", r.text[:300])
        return False, None, took


def report(body, args):
    print("\n=== response ===")
    print("top-level keys:", list(body) if isinstance(body, dict) else type(body).__name__)
    arrays = find_number_arrays(body)
    if not arrays:
        print("no numeric arrays anywhere in the response.")
        print(json.dumps(body, ensure_ascii=False, indent=2)[:2000])
        return None
    print("\nnumeric arrays found (path -> shape):")
    for p, shape in arrays:
        print(f"  {p:50s} {shape}")

    # The action stream should be 2-D and one axis should be the 64 slots.
    picks = [(p, s) for p, s in arrays if len(s) == 2 and 64 in s]
    if not picks:
        picks = [(p, s) for p, s in arrays if len(s) == 2]
    if not picks:
        print("\nnothing that looks like [steps, dims]. Inspect the paths above by hand.")
        return None
    path, shape = picks[0]
    print(f"\nreading actions from {path} (shape {shape})")
    return dig(body, path)


def compare(pred, args):
    """Put predicted and recorded actions side by side."""
    try:
        import numpy as np
    except ImportError:
        print("\nnumpy is not available — skipping the comparison.")
        return
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 2 and pred.shape[0] == 64 and pred.shape[1] != 64:
        pred = pred.T          # want [steps, dims]
    print(f"\npredicted: {pred.shape}")
    live = pred[:, : args.raw_action_dim]
    print(f"first {args.raw_action_dim} dims (raw_action_dim), first 3 steps:")
    for row in live[:3]:
        print("   ", np.array2string(row, precision=4, suppress_small=True))
    print("per-dim range:")
    for d in range(live.shape[1]):
        col = live[:, d]
        print(f"    dim {d}: {col.min():+.4f} .. {col.max():+.4f}   mean {col.mean():+.4f}")
    tail = pred[:, args.raw_action_dim:]
    if tail.size:
        print(f"padding dims {args.raw_action_dim}..{pred.shape[1] - 1}: "
              f"all zero = {bool(np.allclose(tail, 0))}")

    if not args.actions:
        print("\npass --actions to compare against what was recorded.")
        return
    if args.actions.endswith(".npy"):
        true = np.load(args.actions)
    else:
        true = np.loadtxt(args.actions, delimiter=",")
    true = np.atleast_2d(true)
    print(f"\nrecorded: {true.shape}")
    n = min(len(live), len(true))
    d = min(live.shape[1], true.shape[1])
    if n == 0 or d == 0:
        print("nothing overlaps — check the files.")
        return
    print(f"comparing {n} steps x {d} dims")
    print("  dim   corr      pred range              recorded range")
    for k in range(d):
        a, b = live[:n, k], true[:n, k]
        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"  {k:3d}  {c:+.3f}   {a.min():+.4f}..{a.max():+.4f}   "
              f"{b.min():+.4f}..{b.max():+.4f}")
    print("\nCorrelation near +1 means the shape of the motion matches even if the\n"
          "scale or convention differs. Near 0 means the domain does not transfer.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="episode clip (mp4)")
    p.add_argument("--mode", default="inverse_dynamics",
                   choices=["inverse_dynamics", "policy", "forward_dynamics"])
    p.add_argument("--domain", default=None,
                   help=f"embodiment name; tries {', '.join(DOMAINS[:2])} in turn if omitted")
    p.add_argument("--domain-id", type=int, default=None, help="numeric domain instead of a name")
    p.add_argument("--raw-action-dim", type=int, default=8,
                   help="how many of the 64 slots are real for this robot (default 8)")
    p.add_argument("--chunk", type=int, default=None,
                   help="action_chunk_size; must be num_frames or num_frames-1 "
                        "(default: num_frames-1, one action per transition)")
    p.add_argument("--frames", type=int, default=None, help="defaults to the clip's own count")
    p.add_argument("--fps", type=int, default=20, help="must match the source episode")
    p.add_argument("--size", default="832x480")
    p.add_argument("--steps", type=int, default=20, help="keep it low; we want the actions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default="A Franka Panda arm picks up a cube.")
    p.add_argument("--actions", default=None,
                   help="recorded actions to compare against (.npy or .csv, [steps, dims])")
    p.add_argument("--extra", default="", help="extra_params overrides as JSON")
    args = p.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"no such clip: {args.video}")

    n = args.frames or clip_frames(args.video)
    if not n:
        sys.exit("could not count the clip's frames — pass --frames")
    server_path, how = stage_for_server(args.video)
    print(f"clip      : {args.video}  ({n} frames)")
    print(f"server sees: {server_path}  [{how}]")
    print(f"mode      : {args.mode}   fps {args.fps}   steps {args.steps}")

    domains = [args.domain] if args.domain else DOMAINS[:2]
    if args.domain_id is not None:
        domains = [None]

    base = {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": args.prompt,
        "size": args.size,
        "num_frames": n,
        "fps": args.fps,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "generate_sound": "false",
    }

    for domain in domains:
        extra = {
            "action_mode": args.mode,
            "raw_action_dim": args.raw_action_dim,
            "action_fps": args.fps,
            # action.py rejects anything else outright:
            #   "action_chunk_size must equal num_frames - 1 or num_frames".
            # n-1 is one action per transition between frames, which is what
            # inverse dynamics is recovering.
            "action_chunk_size": args.chunk if args.chunk else n - 1,
        }
        if args.domain_id is not None:
            extra["domain_id"] = args.domain_id
        else:
            extra["domain_name"] = domain
        if args.extra.strip():
            extra.update(json.loads(args.extra))

        tag = f"domain={extra.get('domain_name') or extra.get('domain_id')}"

        # The video has to land in multi_modal_data['video']. Which request
        # field does that is the thing we are here to find out, so try the two
        # plausible ones: a server-side path, then a real upload.
        plans = [
            (f"video_reference as a path · {tag}",
             {**base, "video_reference": server_path, "extra_params": json.dumps(extra)}, None),
            (f"input_reference as an upload · {tag}",
             {**base, "extra_params": json.dumps(extra)}, ("input_reference", args.video)),
        ]
        for label, form, files in plans:
            ok, body, _ = attempt(label, form, files)
            if not ok:
                continue
            pred = report(body, args)
            if pred is not None:
                compare(pred, args)
                print(f"\nWorking request shape: {label}")
                return
            print("    reached the server but no action array came back.")

    print("\nNo attempt produced actions. What the errors say is the next lead —\n"
          "a 400 naming multi_modal_data['video'] means the mode was reached but\n"
          "the clip did not arrive; a plain mp4 back means the mode was ignored.")


if __name__ == "__main__":
    main()
