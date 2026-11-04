"""Capture calibration stills from the webcam, with live quality checks.

Mode: 1280x960 MJPG with exposure, gain and white balance locked (the C270 has a
fixed-focus lens and exposes no focus control; a camera that has one gets it locked).

Live view: detected board, estimated tilt, and a 4x4 coverage map whose cells turn
green once a saved photo has corners there. SPACE saves an uncompressed PNG to
data/raw/E0_stills/ only if the whole board is visible and the image is sharp and
still; otherwise the reason is shown. Q or ESC quits. Photos already in the folder
are counted on start and never overwritten.

The tilt shown uses a nominal K (--focal-px, principal point at the image centre),
because the real K is what these photos will produce. calib/relatorio.md gives the
final tilt of each photo.

Run: .venv/bin/python scripts/capture_stills.py
Then: .venv/bin/python calib/calibrate.py --fotos data/raw/E0_stills
"""
import argparse
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import cantos_tabuleiro, carregar_geometria  # noqa: E402
from calib.calibrate import board_sharpness, coverage_grid, tilt_and_normal  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 960, 30
GRID = 4
TILT_TARGET_DEG = 30.0
WINDOW = "capture_stills"
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 255, 255)


@dataclass
class Limits:
    min_sharpness: float = 60.0
    max_speed: float = 0.5   # px per frame at the nominal frame rate
    margin: int = 8          # min distance (px) from corners to the image edge


@dataclass
class Check:
    corners: np.ndarray | None = None
    tilt_deg: float | None = None
    sharpness: float | None = None
    speed: float | None = None
    problems: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems


def nominal_camera(focal_px, size=(WIDTH, HEIGHT)):
    return np.array([[focal_px, 0, size[0] / 2], [0, focal_px, size[1] / 2], [0, 0, 1]])


def estimate_tilt(corners, objp, K):
    ok, rvec, _ = cv2.solvePnP(objp, corners, K, None)
    return tilt_and_normal(rvec)[0] if ok else None


def evaluate(gray, geo, K, limits, previous=None, dt_s=None):
    """Check one frame. `previous` holds the corners of the frame before it."""
    check = Check(corners=cantos_tabuleiro(gray, geo))
    if check.corners is None:
        check.problems.append("tabuleiro incompleto")
        return check
    pts = check.corners.reshape(-1, 2)
    h, w = gray.shape[:2]
    if pts.min() < limits.margin or pts[:, 0].max() > w - 1 - limits.margin \
            or pts[:, 1].max() > h - 1 - limits.margin:
        check.problems.append("tabuleiro colado na borda")
    check.tilt_deg = estimate_tilt(check.corners, geo["objp_tabuleiro"], K)
    check.sharpness = board_sharpness(gray, check.corners)
    if check.sharpness < limits.min_sharpness:
        check.problems.append("imagem borrada")
    if previous is None or not dt_s:
        check.problems.append("sem quadro anterior")
    else:
        shift = float(np.linalg.norm(check.corners - previous, axis=2).mean())
        check.speed = shift / (dt_s * FPS)
        if check.speed > limits.max_speed:
            check.problems.append("em movimento")
    return check


class Session:
    """Photos saved so far, restored from the folder on start."""

    def __init__(self, folder, geo, K, size=(WIDTH, HEIGHT)):
        self.folder, self.size = Path(folder), size
        self.folder.mkdir(parents=True, exist_ok=True)
        self.photos, self.next_index = [], 1
        for path in sorted(self.folder.glob("still_*.png")):
            self.next_index = max(self.next_index, int(re.search(r"\d+", path.stem)[0]) + 1)
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            corners = None if gray is None else cantos_tabuleiro(gray, geo)
            if corners is None:
                print(f"Aviso: tabuleiro não detectado em {path.name}; ela não entra na contagem.")
                continue
            self.add(corners, estimate_tilt(corners, geo["objp_tabuleiro"], K))

    def add(self, corners, tilt_deg):
        self.photos.append({"corners": corners, "tilt_deg": tilt_deg})

    def save(self, frame, check):
        path = self.folder / f"still_{self.next_index:04d}.png"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise SystemExit(f"Could not write {path}")
        self.next_index += 1
        self.add(check.corners, check.tilt_deg)
        return path

    @property
    def n_tilted(self):
        return sum(p["tilt_deg"] is not None and p["tilt_deg"] > TILT_TARGET_DEG for p in self.photos)

    @property
    def grid(self):
        return coverage_grid(self.photos, self.size, cols=GRID, rows=GRID)

    @property
    def coverage(self):
        return float((self.grid > 0).mean())


def draw(frame, check, session, pattern, message):
    """Return an annotated copy; the frame itself stays clean for saving."""
    h, w = frame.shape[:2]
    view = frame.copy()
    tint = view.copy()
    grid = session.grid
    xs = [w * i // GRID for i in range(GRID + 1)]
    ys = [h * i // GRID for i in range(GRID + 1)]
    for r in range(GRID):
        for c in range(GRID):
            if grid[r, c]:
                cv2.rectangle(tint, (xs[c], ys[r]), (xs[c + 1], ys[r + 1]), GREEN, -1)
    view = cv2.addWeighted(tint, 0.25, view, 0.75, 0)
    for x in xs[1:-1]:
        cv2.line(view, (x, 0), (x, h), (255, 255, 255), 1)
    for y in ys[1:-1]:
        cv2.line(view, (0, y), (w, y), (255, 255, 255), 1)
    if check.corners is not None:
        cv2.drawChessboardCorners(view, pattern, check.corners, True)

    def fmt(value, spec):
        return "--" if value is None else format(value, spec)

    lines = [
        (f"Fotos: {len(session.photos)}   >{TILT_TARGET_DEG:.0f} graus: {session.n_tilted}   "
         f"Cobertura: {session.coverage:.0%}", (255, 255, 255)),
        (f"Inclinacao: {fmt(check.tilt_deg, '.0f')} graus   Nitidez: {fmt(check.sharpness, '.0f')}   "
         f"Velocidade: {fmt(check.speed, '.2f')} px/quadro", (255, 255, 255)),
        ("PRONTO: aperte espaco" if check.ok else "NAO SALVA: " + ", ".join(check.problems),
         GREEN if check.ok else RED),
    ]
    if message:
        lines.append((message, YELLOW))
    bar = view[:12 + 30 * len(lines)]   # translucent, so the top cells stay visible
    bar[:] = (bar * 0.4).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(view, text, (10, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return view


class FrameGrabber(threading.Thread):
    """Keep only the newest frame, so slow detection never shows an old image."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.ready = cap, threading.Condition()
        self.frame, self.stamp, self.count, self.running = None, 0.0, 0, True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.ready:
                if not ok:
                    self.running = False
                else:
                    self.frame, self.stamp, self.count = frame, time.monotonic(), self.count + 1
                self.ready.notify_all()

    def next_frame(self, last_count):
        """Block until a frame newer than `last_count` arrives; None if the camera stops."""
        with self.ready:
            self.ready.wait_for(lambda: self.count > last_count or not self.running, timeout=5)
            if self.count <= last_count:
                return None
            return self.frame, self.stamp, self.count


def v4l2(device, *args):
    done = subprocess.run(["v4l2-ctl", "-d", device, *args], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"v4l2-ctl {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def lock_controls(device, exposure, gain, white_balance):
    """Set manual controls and read them back; stop if one did not take effect."""
    available = v4l2(device, "--list-ctrls")
    wanted = {"auto_exposure": 1, "exposure_dynamic_framerate": 0, "exposure_time_absolute": exposure,
              "gain": gain, "white_balance_automatic": 0, "white_balance_temperature": white_balance}
    if "focus_automatic_continuous" in available:
        wanted["focus_automatic_continuous"] = 0
        print("Foco automático desligado.")
    else:
        print("Câmera sem controle de foco (lente de foco fixo).")
    for name, value in wanted.items():   # order matters: auto modes go off first
        if name in available:
            v4l2(device, "-c", f"{name}={value}")
    for name, value in wanted.items():
        if name in available:
            read = re.search(r":\s*(-?\d+)", v4l2(device, "-C", name))
            if read is None or int(read[1]) != value:
                raise SystemExit(f"Control {name} did not lock at {value}.")
    print(f"Travado: exposição {exposure}, ganho {gain}, balanço de branco {white_balance} K.")


def open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    if size != (WIDTH, HEIGHT) or fourcc != "MJPG":
        raise SystemExit(f"Camera mode is {size} {fourcc}, expected {WIDTH}x{HEIGHT} MJPG.")
    return cap


def run(cap, session, geo, K, limits):
    grabber = FrameGrabber(cap)
    grabber.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
    previous, previous_stamp, count = None, None, 0
    message, message_until = "", 0.0
    try:
        while True:
            item = grabber.next_frame(count)
            if item is None:
                print("A câmera parou de enviar quadros.")
                break
            frame, stamp, count = item
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dt_s = None if previous_stamp is None else stamp - previous_stamp
            check = evaluate(gray, geo, K, limits, previous, dt_s)
            previous, previous_stamp = check.corners, stamp

            now = time.monotonic()
            cv2.imshow(WINDOW, draw(frame, check, session, geo["padrao"], message if now < message_until else ""))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if check.ok:
                    path = session.save(frame, check)
                    message = f"Salva: {path.name}"
                    print(f"{path.name}: inclinação {check.tilt_deg:.1f}°, nitidez {check.sharpness:.0f}, "
                          f"cobertura {session.coverage:.0%}")
                else:
                    message = "Nao salvou: " + ", ".join(check.problems)
                message_until = now + 2.0
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        grabber.running = False
        grabber.join(timeout=2)
        cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw/E0_stills")
    ap.add_argument("--exposure", type=int, default=150, help="exposure_time_absolute, in units of 100 us")
    ap.add_argument("--gain", type=int, default=8)
    ap.add_argument("--white-balance", type=int, default=4000, help="colour temperature (K)")
    ap.add_argument("--focal-px", type=float, default=1400, help="nominal focal length for the live tilt")
    ap.add_argument("--min-sharpness", type=float, default=Limits.min_sharpness,
                    help="min Laplacian variance inside the board region")
    ap.add_argument("--max-speed", type=float, default=Limits.max_speed,
                    help="max mean corner speed against the previous frame (px/frame)")
    a = ap.parse_args(argv)

    geo = carregar_geometria()
    K = nominal_camera(a.focal_px)
    session = Session(a.out, geo, K)
    print(f"{len(session.photos)} fotos já em {a.out}.")
    cap = open_camera(a.device)
    try:
        lock_controls(a.device, a.exposure, a.gain, a.white_balance)
        run(cap, session, geo, K, Limits(a.min_sharpness, a.max_speed))
    finally:
        cap.release()
    print(f"Total: {len(session.photos)} fotos, {session.n_tilted} acima de {TILT_TARGET_DEG:.0f}°, "
          f"cobertura {session.coverage:.0%}.")


if __name__ == "__main__":
    main()
