#!/usr/bin/env python3
"""
Build a control video for Cosmos3 transfer from an Isaac Sim episode.

Three input routes, in order of how good the result will be:

  --mode depth --depth-dir <dir>     Isaac Sim ground-truth depth (best).
                                     Reads *.npy / *.exr / 16-bit *.png written
                                     by a Replicator distance_to_camera writer.

  --mode seg   --seg-dir <dir>       Isaac Sim semantic segmentation frames.
                                     Colour ids are already stable, so this is
                                     a straight re-encode.

  --mode edge  --rgb <mp4>           Canny edges off the RGB clip. Needs no
                                     extra data and no model download, so this
                                     is the one to test the pipeline with today.

Why the depth path is fussy
---------------------------
Depth is normalised with ONE range for the whole episode, taken from robust
percentiles. Per-frame normalisation is the usual mistake: it makes the depth
scale breathe, and the generated video flickers because the model reads a
moving surface where the table should be still.

Metric depth from the simulator has near = small. Cosmos was trained on
inverse-depth-style controls where near = bright, so the values are inverted by
default. --no-invert if your writer already emits inverse depth.

Usage
-----
  python make_control.py --mode edge  --rgb episode_000.mp4 --out ctrl_edge.mp4
  python make_control.py --mode depth --depth-dir ep000/depth --fps 24 \
                         --out ctrl_depth.mp4
  python make_control.py --mode seg   --seg-dir ep000/semantic --out ctrl_seg.mp4

Deps:  pip install imageio imageio-ffmpeg numpy    (opencv-python for --mode edge)
"""

import argparse
import glob
import os
import sys

import numpy as np


def load_depth_frame(path):
    """One depth frame as float32, whatever the writer produced."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
    elif ext in (".png", ".tif", ".tiff"):
        import imageio.v3 as iio
        a = iio.imread(path).astype(np.float32)
    elif ext == ".exr":
        try:
            import imageio.v3 as iio
            a = iio.imread(path).astype(np.float32)
        except Exception as e:
            sys.exit(f"Cannot read {path} ({e}). pip install imageio[openexr]")
    else:
        sys.exit(f"Unsupported depth format: {path}")
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:            # some writers store depth in the first channel
        a = a[..., 0]
    return a


def depth_range(files, lo_pct, hi_pct, sample=40):
    """Robust min/max over the whole episode, from a subsample of frames.

    Sampling keeps this fast on long episodes; the range only needs to be
    stable, not exact.
    """
    step = max(1, len(files) // sample)
    vals = []
    for p in files[::step]:
        a = load_depth_frame(p).ravel()
        a = a[np.isfinite(a)]
        # Isaac Sim writes 0 or a huge sentinel for "no hit" (sky / outside the
        # far plane). Both would swallow the useful range.
        a = a[(a > 0) & (a < 1e6)]
        if a.size:
            vals.append(np.random.choice(a, size=min(a.size, 50000), replace=False))
    if not vals:
        sys.exit("No usable depth values found.")
    all_v = np.concatenate(vals)
    return float(np.percentile(all_v, lo_pct)), float(np.percentile(all_v, hi_pct))


def to_gray8(a, lo, hi, invert):
    a = np.nan_to_num(a, nan=hi, posinf=hi, neginf=lo)
    a = np.clip(a, lo, hi)
    norm = (a - lo) / max(hi - lo, 1e-6)
    if invert:
        norm = 1.0 - norm          # near = bright
    return (norm * 255.0).astype(np.uint8)


def edge_frame(rgb, lo, hi, thickness):
    """Canny if OpenCV is around, Sobel otherwise so this never hard-fails."""
    try:
        import cv2
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        e = cv2.Canny(g, lo, hi)
        if thickness > 1:
            e = cv2.dilate(e, np.ones((thickness, thickness), np.uint8))
        return e
    except ImportError:
        g = rgb.astype(np.float32).mean(axis=2)
        gx = np.zeros_like(g); gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        m = np.hypot(gx, gy)
        thr = np.percentile(m, 92)
        return ((m > thr) * 255).astype(np.uint8)


def even(v):
    """h264 needs even dimensions."""
    return v - (v % 2)


def write_video(frames, out, fps):
    """h264 mp4, with whichever imageio video backend this machine has.

    imageio v3 dispatches to pyav when the `av` package is importable and to
    imageio-ffmpeg otherwise, and the two spell the same two things differently:
    pyav takes out_pixel_format and has no way to pass encoder flags through
    imwrite at all, while imageio-ffmpeg takes pixelformat plus output_params.
    Asking for the wrong one is a TypeError, so try both.

    imageio-ffmpeg goes first because it is the one that accepts -crf. Quality is
    not cosmetic here: the depth silhouettes are the geometry signal Cosmos reads,
    and default-quality h264 rings around exactly those edges.
    """
    import imageio.v3 as iio
    arr = np.stack(frames)
    h, w = arr.shape[1:3]
    if (h, w) != (even(h), even(w)):
        sys.exit(f"Frame size {w}x{h} is odd; crop or resize to even dimensions.")

    attempts = [
        ("FFMPEG", dict(fps=fps, codec="libx264", pixelformat="yuv420p",
                        output_params=["-crf", "12"])),
        ("pyav", dict(fps=fps, codec="libx264", out_pixel_format="yuv420p")),
    ]
    problems = []
    for plugin, kw in attempts:
        try:
            iio.imwrite(out, arr, plugin=plugin, **kw)
        except Exception as e:
            problems.append(f"{plugin}: {type(e).__name__}: {e}")
            continue
        note = "" if plugin == "FFMPEG" else (
            "  [!] pyav cannot take -crf through imwrite, so this used the "
            "encoder default. For near-lossless control videos: pip install "
            "imageio-ffmpeg")
        print(f"wrote {out}  ({len(frames)} frames, {w}x{h}, {fps}fps, via {plugin}){note}")
        return
    sys.exit("Could not write the mp4 with either video backend:\n  "
             + "\n  ".join(problems)
             + "\nInstall one:  pip install imageio-ffmpeg   (or)  pip install av")


def frame_files(d, exts):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, f"*{e}"))
    if not files:
        sys.exit(f"No {'/'.join(exts)} files in {d}")
    # Replicator names frames with zero-padded counters, so a plain sort is the
    # temporal order.
    return sorted(files)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["depth", "seg", "edge"])
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=24)

    p.add_argument("--rgb", help="source mp4, for --mode edge")
    p.add_argument("--depth-dir", help="folder of depth frames")
    p.add_argument("--seg-dir", help="folder of segmentation frames")

    p.add_argument("--lo-pct", type=float, default=1.0,
                   help="lower percentile for the global depth range")
    p.add_argument("--hi-pct", type=float, default=99.0)
    p.add_argument("--no-invert", action="store_true",
                   help="keep far = bright (use if input is already inverse depth)")

    p.add_argument("--canny-lo", type=int, default=80)
    p.add_argument("--canny-hi", type=int, default=180)
    p.add_argument("--edge-thickness", type=int, default=2,
                   help="dilate edges; 1 leaves them hairline")

    p.add_argument("--max-frames", type=int,
                   help="truncate (720p transfer tops out around 300 frames)")
    args = p.parse_args()

    import imageio.v3 as iio
    frames = []

    if args.mode == "depth":
        if not args.depth_dir:
            p.error("--mode depth needs --depth-dir")
        files = frame_files(args.depth_dir, [".npy", ".png", ".exr", ".tiff", ".tif"])
        if args.max_frames:
            files = files[:args.max_frames]
        lo, hi = depth_range(files, args.lo_pct, args.hi_pct)
        print(f"global depth range: {lo:.3f} .. {hi:.3f} "
              f"({'inverted, near=bright' if not args.no_invert else 'as-is'})")
        for f in files:
            g = to_gray8(load_depth_frame(f), lo, hi, not args.no_invert)
            frames.append(np.repeat(g[:, :, None], 3, axis=2))

    elif args.mode == "seg":
        if not args.seg_dir:
            p.error("--mode seg needs --seg-dir")
        files = frame_files(args.seg_dir, [".png", ".jpg", ".jpeg"])
        if args.max_frames:
            files = files[:args.max_frames]
        for f in files:
            a = iio.imread(f)
            if a.ndim == 2:                       # id map, not colourised
                a = np.repeat(a.astype(np.uint8)[:, :, None], 3, axis=2)
            frames.append(np.ascontiguousarray(a[..., :3]))

    else:  # edge
        if not args.rgb:
            p.error("--mode edge needs --rgb")
        for i, fr in enumerate(iio.imiter(args.rgb)):
            if args.max_frames and i >= args.max_frames:
                break
            e = edge_frame(fr[..., :3], args.canny_lo, args.canny_hi,
                           args.edge_thickness)
            frames.append(np.repeat(e[:, :, None], 3, axis=2))

    if not frames:
        sys.exit("No frames produced.")
    if len(frames) > 121:
        # Not a problem to fix here: transfer generates in chunks, so a long
        # control clip is normal (the paper uses 93 frames per chunk on episodes
        # longer than this). Splitting the episode would be the *wrong* fix — it
        # is the frame count matching the action labels that has to be preserved.
        print(f"note: {len(frames)} frames, longer than one chunk (121 max). "
              f"Transfer will generate this in chunks — set "
              f"num_video_frames_per_chunk in the GUI's Transfer tab (93 is what "
              f"the paper used). Do not split the episode to make it fit.")
    write_video(frames, args.out, args.fps)
    print("Frame count must stay identical to the RGB episode so the LeRobot "
          "action labels still line up.")


if __name__ == "__main__":
    sys.exit(main())
