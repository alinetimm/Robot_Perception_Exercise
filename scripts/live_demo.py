"""Live demo for the video: marker axes, 3-sigma ellipse, error and NEES on screen.

The state goes from DETECTADO to RASTREADO when the marker is covered, and the ellipse
grows with the time without detection. The pose never uses the board. A second thread
detects the board and compares it with the estimate of that same frame, so the error
and the NEES on screen are correct even though they update less often than the image.

Keys: R starts and stops a recording of the screen to video/, Q or ESC quits.
Run: .venv/bin/python scripts/live_demo.py
"""
import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from alvo.referencia import carregar_geometria_medida  # noqa: E402
from capture_stills import (FPS, GREEN, HEIGHT, RED, WIDTH, YELLOW, FrameGrabber, banner, find_camera,  # noqa: E402
                            lock_controls, open_camera)
from src.live import DETECTED, LOST, TRACKED, LiveEstimator, reference_check, tip_ellipse  # noqa: E402
from src.pose import load_camera, marker_model, tilt_deg  # noqa: E402

WINDOW = "live_demo"
WHITE = (255, 255, 255)
STATE_COLOR = {DETECTED: GREEN, TRACKED: YELLOW, LOST: RED}
AXIS_MM = 50.0
TIP_MM = 100.0
CHECK_MAX_AGE_S = 1.0


def load_models():
    speed_model = json.loads((ROOT / "results/nees.json").read_text(encoding="utf-8"))["speed_model"]
    q = json.loads((ROOT / "results/occlusion.json").read_text(encoding="utf-8"))["methods"]["ring"]["q_sd_per_sqrt_frame"]
    growth = np.r_[np.radians(q["rot_deg"]), q["trans_mm"]] ** 2
    return speed_model, growth


class ReferenceWorker(threading.Thread):
    """Check the newest estimate against the board, dropping frames it cannot keep up with."""

    def __init__(self, geo, model, camera, speed_model):
        super().__init__(daemon=True)
        self.args = (geo, model, camera, speed_model)
        self.ready = threading.Condition()
        self.job = None
        self.result, self.result_stamp = None, 0.0
        self.running = True

    def submit(self, gray, estimate):
        with self.ready:
            self.job = (gray, estimate)
            self.ready.notify()

    def latest(self):
        if time.monotonic() - self.result_stamp > CHECK_MAX_AGE_S:
            return None
        return self.result

    def run(self):
        while self.running:
            with self.ready:
                self.ready.wait_for(lambda: self.job is not None or not self.running, timeout=0.5)
                job, self.job = self.job, None
            if job is not None:
                self.result, self.result_stamp = reference_check(job[0], job[1], *self.args), time.monotonic()


def draw(frame, estimate, camera, check, gain, rate, recording):
    view = frame
    if estimate.R is not None:
        rvec = cv2.Rodrigues(estimate.R)[0]
        cv2.polylines(view, [estimate.corners.astype(np.int32)], True, STATE_COLOR[estimate.state], 2, cv2.LINE_AA)
        cv2.drawFrameAxes(view, camera.K, camera.dist, rvec, estimate.t.reshape(3, 1), AXIS_MM, 2)
        centre, axes, angle = tip_ellipse(estimate.R, estimate.t, estimate.cov, camera, TIP_MM, 3.0, gain)
        base = cv2.projectPoints(np.zeros((1, 3)), rvec, estimate.t, camera.K, camera.dist)[0].ravel()
        cv2.line(view, tuple(np.rint(base).astype(int)), tuple(np.rint(centre).astype(int)), WHITE, 1, cv2.LINE_AA)
        limit = max(frame.shape)
        cv2.ellipse(view, tuple(np.rint(centre).astype(int)),
                    tuple(int(min(a, limit)) + 1 for a in axes), angle, 0, 360, STATE_COLOR[estimate.state], 2, cv2.LINE_AA)

    title = estimate.state
    if estimate.state == TRACKED:
        title += f"  ha {estimate.frames_lost / FPS:.1f} s"
    lines = [(title, STATE_COLOR[estimate.state])]
    if estimate.R is not None:
        sd = np.sqrt(np.diag(estimate.cov))
        lines.append((f"Inclinacao {tilt_deg(estimate.R):.0f} graus   Distancia {np.linalg.norm(estimate.t) / 1000:.2f} m   "
                      f"Sigma rot {np.degrees(np.linalg.norm(sd[:2])):.2f} graus   Sigma prof {sd[5]:.1f} mm", WHITE))
    if check is not None:
        lines.append((f"Erro {check['rot_deg']:.2f} graus, {check['trans_mm']:.1f} mm   NEES {check['nees']:.1f}", WHITE))
    else:
        lines.append(("Erro e NEES: sem tabuleiro na imagem", WHITE))
    lines.append((f"Elipse 3 sigma na ponta da normal (ganho {gain:g}x)   {rate:.0f} Hz"
                  + ("   GRAVANDO" if recording else ""), RED if recording else WHITE))
    banner(view, lines)
    return view


def next_recording(folder):
    numbers = [int(m[1]) for p in folder.glob("demo_take*.mp4") if (m := re.search(r"take(\d+)", p.stem))]
    return folder / f"demo_take{max(numbers, default=0) + 1:02d}.mp4"


class ScreenRecorder:
    """Writes at the nominal rate, repeating frames, so the video keeps the real time."""

    def __init__(self, path):
        self.path = path
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        if not self.writer.isOpened():
            raise SystemExit(f"Could not write {path}")
        self.start, self.written = time.monotonic(), 0

    def add(self, frame):
        due = int((time.monotonic() - self.start) * FPS) + 1
        while self.written < due:
            self.writer.write(frame)
            self.written += 1

    def close(self):
        self.writer.release()
        print(f"{self.path.name}: {self.written / FPS:.1f} s")


def run(cap, estimator, geo, model, camera, speed_model, gain, folder):
    grabber = FrameGrabber(cap)
    worker = ReferenceWorker(geo, model, camera, speed_model)
    grabber.start()
    worker.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
    count, recorder, rate, last = 0, None, 0.0, time.monotonic()
    try:
        while True:
            item = grabber.next_frame(count)
            if item is None:
                print("A câmera parou de enviar quadros.")
                break
            frame, stamp, count = item
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            estimate = estimator.step(gray, stamp)
            if estimate.R is not None:
                worker.submit(gray, estimate)
            now = time.monotonic()
            rate = 0.9 * rate + 0.1 / max(now - last, 1e-3)
            last = now
            view = draw(frame.copy(), estimate, camera, worker.latest() if estimate.R is not None else None,
                        gain, rate, recorder is not None)
            cv2.imshow(WINDOW, view)
            if recorder is not None:
                recorder.add(view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if recorder is None:
                    recorder = ScreenRecorder(next_recording(folder))
                else:
                    recorder.close()
                    recorder = None
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if recorder is not None:
            recorder.close()
        grabber.running = worker.running = False
        grabber.join(timeout=2)
        worker.join(timeout=2)
        cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="video device (default: first camera found)")
    ap.add_argument("--out", type=Path, default=ROOT / "video")
    ap.add_argument("--exposure", type=int, default=150, help="exposure_time_absolute, in units of 100 us")
    ap.add_argument("--gain", type=int, default=8)
    ap.add_argument("--white-balance", type=int, default=4000, help="colour temperature (K)")
    ap.add_argument("--ellipse-gain", type=float, default=5.0, help="magnification of the ellipse, shown on screen")
    a = ap.parse_args(argv)

    geo = carregar_geometria_medida()
    model = marker_model(geo["objp_marcador"])
    camera = load_camera(ROOT / "calib/camera.yaml")
    speed_model, growth = load_models()
    a.out.mkdir(parents=True, exist_ok=True)
    device = a.device or find_camera()
    print(f"Câmera: {device}")
    cap = open_camera(device)
    try:
        lock_controls(device, a.exposure, a.gain, a.white_balance)
        run(cap, LiveEstimator(geo, model, camera, speed_model, growth), geo, model, camera, speed_model,
            a.ellipse_gain, a.out)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
