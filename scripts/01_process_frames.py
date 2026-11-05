"""Per-frame poses and marker errors for the evaluation videos.

Reads data/raw/, calib/camera.yaml and calib/alvo_medido.json; writes one Parquet
file per video to data/proc/frames/. Long videos are split into chunks that run in
parallel. Each chunk uses one OpenCV thread, so the output is deterministic.

Run: .venv/bin/python scripts/01_process_frames.py [E1 E2 ...]
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import carregar_geometria_medida  # noqa: E402
from src.geometry import log_so3  # noqa: E402
from src.pose import load_camera, marker_model, process_frame  # noqa: E402

RAW = ROOT / "data/raw"
OUT = ROOT / "data/proc/frames"
VIDEOS = {"E1": "E1_take05.mjpg", "E1_far": "E1_take04.mjpg", "E1_webm": "E1.webm", "E2": "E2.webm",
          "E3": "E3.webm", "E5": "E5.webm"}
CHUNK_FRAMES = 1500


def frame_times(path):
    """Capture times from the recorder's CSV; None if the video carries its own."""
    table = path.with_suffix(".csv")
    if not table.exists():
        return None
    with open(table, encoding="utf-8") as f:
        return [float(r["time_s"]) for r in csv.DictReader(f)]


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def flatten(row):
    """Turn the arrays of one frame into named columns."""
    out = {k: row[k] for k in ("frame", "time_s", "board_ok", "marker_ok")}
    for key in ("ref_rms_px", "tilt_deg", "distance_mm", "m_rms_px", "m2_rms_px", "ippe_ratio"):
        out[key] = row.get(key, np.nan)
    for prefix in ("ref", "m", "m2"):
        if f"{prefix}_R" in row:
            r, t = log_so3(row[f"{prefix}_R"]), row[f"{prefix}_t"]
        else:
            r = t = np.full(3, np.nan)
        for i, axis in enumerate("xyz"):
            out[f"{prefix}_r{axis}"], out[f"{prefix}_t{axis}"] = r[i], t[i]
    for prefix, key in (("err", "err"), ("err2", "err2")):
        e = row.get(key, np.full(6, np.nan))
        out[f"{prefix}_rot_deg"] = float(np.degrees(np.linalg.norm(e[:3])))
        out[f"{prefix}_trans_mm"] = float(np.linalg.norm(e[3:]))
        for i, axis in enumerate("xyz"):
            out[f"{prefix}_dtheta_{axis}_deg"] = float(np.degrees(e[i]))
            out[f"{prefix}_dt_{axis}_mm"] = e[3 + i]
    for key, name, n in (("marker_corners", "mc", 4), ("board_corners", "bc", 40)):
        c = row[key]
        values = np.full((n, 2), np.nan) if c is None else c.reshape(-1, 2)
        for i in range(n):
            out[f"{name}{i}_x"], out[f"{name}{i}_y"] = float(values[i, 0]), float(values[i, 1])
    return out


def process_chunk(job):
    path, start, stop = job
    cv2.setNumThreads(1)
    geo = carregar_geometria_medida()
    model = marker_model(geo["objp_marcador"])
    camera = load_camera(ROOT / "calib/camera.yaml")
    times = frame_times(path)
    cap = cv2.VideoCapture(str(path))
    rows = []
    index = -1
    while index + 1 < stop and cap.grab():
        index += 1
        if index < start:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        row = process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), geo, model, camera)
        row["frame"] = index
        row["time_s"] = times[index] if times else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        rows.append(flatten(row))
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", default=list(VIDEOS), help=f"subset of {list(VIDEOS)}")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    jobs, owners = [], []
    for name in a.videos:
        path = RAW / VIDEOS[name]
        total = count_frames(path)
        print(f"{name}: {path.name}, {total} frames", flush=True)
        for start in range(0, total, CHUNK_FRAMES):
            jobs.append((path, start, min(start + CHUNK_FRAMES, total)))
            owners.append(name)
    tables = {name: [] for name in a.videos}
    with ProcessPoolExecutor(a.workers) as pool:
        for name, rows in zip(owners, pool.map(process_chunk, jobs)):
            tables[name] += rows
    for name, rows in tables.items():
        df = pd.DataFrame(rows).sort_values("frame", ignore_index=True)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        both = df[df.board_ok & df.marker_ok]
        print(f"{name}: {len(df)} frames, board {df.board_ok.mean():.1%}, marker {df.marker_ok.mean():.1%}, "
              f"both {len(both) / len(df):.1%}, median error {both.err_rot_deg.median():.2f} deg, "
              f"{both.err_trans_mm.median():.1f} mm", flush=True)


if __name__ == "__main__":
    main()
