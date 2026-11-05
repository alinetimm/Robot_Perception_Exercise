"""Access to the per-frame tables written by scripts/01_process_frames.py."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data/proc/frames"


def load(name):
    return pd.read_parquet(FRAMES / f"{name}.parquet")


def corners(df, prefix, n):
    """Corner columns as an array of shape (frames, n, 2)."""
    columns = [f"{prefix}{i}_{axis}" for i in range(n) for axis in "xy"]
    return df[columns].to_numpy(float).reshape(len(df), n, 2)


def marker_speed(df):
    """Mean marker corner motion against the previous frame, in px per 1/30 s.

    Uses only past frames and the marker, so it is available during a real landing.
    A frame with no marker right before it inherits the next known value.
    """
    points = corners(df, "mc", 4)
    step = np.linalg.norm(points[1:] - points[:-1], axis=2).mean(axis=1)
    elapsed = np.diff(df.time_s.to_numpy())
    speed = np.full(len(df), np.nan)
    speed[1:] = step / np.maximum(elapsed * 30, 1e-6)
    speed[1:][np.diff(df.frame.to_numpy()) != 1] = np.nan
    return pd.Series(speed, index=df.index).bfill().ffill().fillna(0.0).to_numpy()


def pose(row, prefix):
    R = cv2.Rodrigues(np.array([row[f"{prefix}_r{a}"] for a in "xyz"], float))[0]
    return R, np.array([row[f"{prefix}_t{a}"] for a in "xyz"], float)
