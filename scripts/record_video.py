"""Record the evaluation videos E1..E5 with live guidance for each protocol of PLANO.md.

The camera JPEG of each frame is written as it arrives, so the video has no second
compression. Output: data/raw/<scene>_take<NN>.mjpg, which cv2.VideoCapture reads,
plus <scene>_take<NN>.csv with the time of each frame (use it, not the file's fps).
Existing takes are never overwritten.

Keys: SPACE starts and stops the recording, Q or ESC quits. Exposure, gain and white
balance are locked as in capture_stills.py. The tilt and distance shown use a nominal
K, so they are guidance, not results.

Run: .venv/bin/python scripts/record_video.py E1
"""
import argparse
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from alvo.referencia import cantos_marcador, carregar_geometria  # noqa: E402
from calib.calibrate import SIDE_TILT_DEG, SIDES, tilt_side  # noqa: E402
from capture_stills import (  # noqa: E402
    FPS, GREEN, RED, WINDOW, YELLOW, FrameGrabber, Limits, banner, evaluate, find_camera, lock_controls,
    nominal_camera, open_camera)

WHITE = (255, 255, 255)
WINDOW_S = 5.0            # recent window for detection rates
SLOW_PX = 3.0             # px/frame: above this, the motion is too fast for E2 and E4
HOLD_DRIFT_PX = 3.0       # max mean corner shift from the start of a hold. Frame-to-frame speed
                          # cannot be used: detection noise alone reaches 0.6 px/frame on a resting
                          # board, while the best 10 s windows of real takes drift 0.8 to 2.3 px
DROPOUT_S = 0.5           # a detection gap shorter than this does not restart a hold
GLARE_LEVEL = 250
SIDE_NAME = {"esquerda": "esq", "direita": "dir", "topo": "cima", "base": "baixo"}


@dataclass
class Sample:
    """What one analysed frame showed."""
    dt: float
    board: bool
    marker: bool
    tilt: float | None = None
    distance_m: float | None = None
    speed: float | None = None
    side: str | None = None
    glare: float = 0.0
    corners: np.ndarray | None = None


class Guide:
    """Common state: elapsed time and detection rates over the last seconds."""
    target_s = 60.0
    title = ""

    def __init__(self):
        self.elapsed = 0.0
        self.recent = deque()

    def update(self, s):
        self.elapsed += s.dt
        self.recent.append(s)
        total = sum(r.dt for r in self.recent)
        while total > WINDOW_S and len(self.recent) > 1:
            total -= self.recent.popleft().dt
        self.observe(s)

    def observe(self, s):
        pass

    def rate(self, attribute):
        if not self.recent:
            return 0.0
        return float(np.mean([getattr(r, attribute) for r in self.recent]))

    def common_lines(self):
        lines = []
        if self.recent and self.rate("board") < 0.8:
            lines.append(("Tabuleiro sumindo: ele e a referencia, mantenha inteiro na imagem", RED))
        return lines

    @property
    def done(self):
        return self.elapsed >= self.target_s

    def lines(self):
        return []

    def summary(self):
        return f"{self.elapsed:.0f} s analisados"


class StillPoses(Guide):
    """E1: five poses held still for 10 s each. The board rests on a support and the
    camera moves between poses: held by hand, the board drifts by more than the limit."""
    title = "E1: placa APOIADA, sem tocar. Entre as poses, mova a WEBCAM"
    poses = [(0.5, 0), (0.8, 30), (1.0, 0), (1.2, 45), (0.6, 60)]
    hold_s = 10.0
    distance_tol_m, tilt_tol = 0.12, 12.0

    def __init__(self):
        super().__init__()
        self.index, self.held = 0, 0.0
        self.anchor, self.drift, self.gap, self.moved_notice = None, 0.0, 0.0, 0.0

    def restart_hold(self):
        self.anchor, self.held, self.drift, self.gap = None, 0.0, 0.0, 0.0

    def on_target(self, s):
        distance, tilt = self.poses[self.index]
        return (s.distance_m is not None and abs(s.distance_m - distance) <= self.distance_tol_m
                and abs(s.tilt - tilt) <= self.tilt_tol)

    def adjustment(self, s):
        """Instruction to reach the current pose from where the board is now."""
        distance, tilt = self.poses[self.index]
        if abs(s.distance_m - distance) > self.distance_tol_m:
            verb = "Aproxime" if s.distance_m > distance else "Afaste"
            return f"{verb} a webcam {abs(s.distance_m - distance):.2f} m (agora {s.distance_m:.2f} m)"
        side = "mais de lado" if s.tilt < tilt else "mais de frente"
        return f"Leve a webcam para ver a placa {side} (agora {s.tilt:.0f} graus)"

    def observe(self, s):
        self.moved_notice = max(0.0, self.moved_notice - s.dt)
        if self.done:
            return
        if not (s.board and s.marker):
            self.gap += s.dt
            if self.gap > DROPOUT_S:
                self.restart_hold()
            return
        self.gap = 0.0
        if self.anchor is None:
            if self.on_target(s):
                self.anchor = s.corners
            return
        self.drift = float(np.linalg.norm(s.corners - self.anchor, axis=2).mean())
        if self.drift > HOLD_DRIFT_PX:
            self.restart_hold()
            self.moved_notice = 2.0
            return
        self.held += s.dt
        if self.held >= self.hold_s:
            self.index += 1
            self.restart_hold()

    @property
    def done(self):
        return self.index >= len(self.poses)

    def lines(self):
        if self.done:
            return [("5 poses concluidas: aperte espaco para parar", GREEN)]
        distance, tilt = self.poses[self.index]
        s = self.recent[-1] if self.recent else None
        out = [(f"Pose {self.index + 1}/5: {distance:.1f} m e {tilt} graus", YELLOW)]
        if s is None or not s.board:
            return out + [("Tabuleiro nao visivel", RED)]
        if not s.marker:
            return out + [("Marcador nao visivel", RED)]
        if self.moved_notice:
            return out + [(f"Moveu mais de {HOLD_DRIFT_PX:.0f} px: contagem reiniciada. Nao toque na placa", RED)]
        if self.anchor is None:
            if not self.on_target(s):
                return out + [(self.adjustment(s), RED)]
            return out + [("Pose certa: nao toque, a contagem vai comecar", YELLOW)]
        return out + [(f"PARADO: {self.held:.0f}/{self.hold_s:.0f} s, nao toque "
                       f"(moveu {self.drift:.1f} de {HOLD_DRIFT_PX:.0f} px)", GREEN)]

    def summary(self):
        return f"{self.index}/5 poses completas"


class SlowMotion(Guide):
    """E2: slow rocking to about 30 degrees in every direction."""
    title = "E2: balance DEVAGAR, ate cerca de 30 graus para os 4 lados"
    target_s = 120.0
    band = (20.0, 40.0)

    def __init__(self):
        super().__init__()
        self.sides = dict.fromkeys(SIDES, 0.0)
        self.fast = 0.0

    def observe(self, s):
        if s.tilt is not None and s.tilt > SIDE_TILT_DEG and self.band[0] <= s.tilt <= self.band[1]:
            self.sides[s.side] += s.dt
        if s.speed is not None and s.speed > SLOW_PX:
            self.fast += s.dt

    def lines(self):
        s = self.recent[-1] if self.recent else None
        out = [(f"Tempo: {self.elapsed:.0f}/{self.target_s:.0f} s   Referencia: {self.rate('board'):.0%}   "
                f"Marcador: {self.rate('marker'):.0%}", WHITE)]
        out.append(("Lados (s): " + "  ".join(f"{SIDE_NAME[k]} {self.sides[k]:.0f}" for k in SIDES), WHITE))
        if s is not None and s.speed is not None and s.speed > SLOW_PX:
            out.append((f"Rapido demais ({s.speed:.1f} px/quadro): mais devagar", RED))
        else:
            short = min(SIDES, key=lambda k: self.sides[k])
            out.append((f"Bom ritmo. Agora: lado {SIDE_NAME[short]} mais perto da camera", GREEN))
        return out

    def summary(self):
        share = self.fast / max(self.elapsed, 1e-9)
        return f"{self.elapsed:.0f} s, {share:.0%} do tempo rapido demais"


class StrongTilt(Guide):
    """E3: strong tilts and passes through the frontal pose."""
    title = "E3: inclinacoes FORTES (60 a 75 graus) e passagens pela frontal"
    target_s = 60.0
    strong_target_s = 15.0
    frontal_target = 5

    def __init__(self):
        super().__init__()
        self.strong, self.frontal, self.in_frontal = 0.0, 0, False

    def observe(self, s):
        if s.tilt is None:
            return
        if s.board and s.marker and 60 <= s.tilt <= 75:
            self.strong += s.dt
        if s.tilt < 8 and not self.in_frontal:
            self.frontal += 1
        self.in_frontal = s.tilt < 15

    @property
    def done(self):
        return self.strong >= self.strong_target_s and self.frontal >= self.frontal_target

    def lines(self):
        s = self.recent[-1] if self.recent else None
        out = [(f"Tempo em 60-75 graus: {self.strong:.0f}/{self.strong_target_s:.0f} s   "
                f"Frontais: {self.frontal}/{self.frontal_target}", WHITE)]
        if s is None or s.tilt is None:
            return out + [("Tabuleiro perdido: reduza a inclinacao ou afaste a placa", RED)]
        if s.tilt > 75:
            return out + [(f"{s.tilt:.0f} graus: passou de 75, volte um pouco", RED)]
        if 60 <= s.tilt <= 75:
            if s.marker:
                return out + [(f"{s.tilt:.0f} graus: faixa certa, segure devagar", GREEN)]
            return out + [(f"{s.tilt:.0f} graus: marcador nao detectado", RED)]
        return out + [(f"{s.tilt:.0f} graus: incline ate 60-75, depois passe pela frontal", YELLOW)]

    def summary(self):
        return f"{self.strong:.0f} s entre 60 e 75 graus, {self.frontal} passagens frontais"


class Glare(SlowMotion):
    """E4: the E2 motion under direct light, with a film over the target."""
    title = "E4: movimento do E2 com BRILHO da luminaria sobre o filme"
    target_s = 60.0
    glare_target = 0.02

    def __init__(self):
        super().__init__()
        self.glare_s = 0.0

    def observe(self, s):
        super().observe(s)
        if s.glare >= self.glare_target:
            self.glare_s += s.dt

    def common_lines(self):
        if self.recent and self.rate("board") < 0.8:
            return [("Referencia falhando: estique o filme ou mude a luz. Sem tabuleiro o E4 nao serve", RED)]
        return []

    def lines(self):
        s = self.recent[-1] if self.recent else None
        out = [(f"Tempo: {self.elapsed:.0f}/{self.target_s:.0f} s   Referencia: {self.rate('board'):.0%}   "
                f"Marcador: {self.rate('marker'):.0%}   Com brilho: {self.glare_s:.0f} s", WHITE)]
        if s is not None and s.speed is not None and s.speed > SLOW_PX:
            return out + [("Rapido demais: mais devagar", RED)]
        if s is None or s.glare < self.glare_target:
            return out + [("Sem brilho no alvo: aproxime a luminaria ou mude o angulo", YELLOW)]
        return out + [(f"Brilho presente ({s.glare:.0%} do tabuleiro). Continue devagar", GREEN)]

    def summary(self):
        return f"{self.elapsed:.0f} s, referencia em {self.rate('board'):.0%} no final, {self.glare_s:.0f} s com brilho"


class Occlusion(Guide):
    """E5: cover the marker for 1 to 2 s, several times, keeping the board visible."""
    title = "E5: cubra SO o marcador por 1 a 2 s, varias vezes"
    target_s = 60.0
    occlusion_target = 8
    min_s, max_s = 1.0, 2.5

    def __init__(self):
        super().__init__()
        self.count, self.covered, self.board_lost = 0, 0.0, False

    def observe(self, s):
        if not s.marker:
            self.covered += s.dt
            self.board_lost |= not s.board
            return
        if self.min_s <= self.covered <= self.max_s and not self.board_lost:
            self.count += 1
        self.covered, self.board_lost = 0.0, False

    @property
    def done(self):
        return self.count >= self.occlusion_target

    def lines(self):
        out = [(f"Oclusoes boas: {self.count}/{self.occlusion_target}   Tempo: {self.elapsed:.0f} s", WHITE)]
        if self.covered == 0:
            return out + [("Marcador visivel: cubra com a mao", YELLOW)]
        if self.board_lost:
            return out + [("A mao cobriu o tabuleiro: esta oclusao nao vale", RED)]
        if self.covered > self.max_s:
            return out + [(f"Coberto ha {self.covered:.1f} s: descubra agora", RED)]
        if self.covered < self.min_s:
            return out + [(f"Coberto: {self.covered:.1f} s, segure", YELLOW)]
        return out + [(f"Coberto: {self.covered:.1f} s: pode descobrir", GREEN)]

    def summary(self):
        return f"{self.count} oclusoes de {self.min_s:g} a {self.max_s:g} s"


SCENES = {"E1": StillPoses, "E2": SlowMotion, "E3": StrongTilt, "E4": Glare, "E5": Occlusion}


def next_take(folder, scene):
    numbers = [int(m[1]) for p in folder.glob(f"{scene}_take*.mjpg")
               if (m := re.search(r"take(\d+)", p.stem))]
    return folder / f"{scene}_take{max(numbers, default=0) + 1:02d}.mjpg"


class Recorder(FrameGrabber):
    """Write every camera JPEG as it arrives; the viewer only gets the newest one."""

    def __init__(self, cap):
        super().__init__(cap)
        self.lock = threading.Lock()
        self.video = self.times = None
        self.written = self.late = 0
        self.first = self.last = None

    def start_take(self, path):
        with self.lock:
            self.video = open(path, "wb")
            self.times = open(path.with_suffix(".csv"), "w", encoding="utf-8")
            self.times.write("frame,time_s\n")
            self.written = self.late = 0
            self.first = self.last = None

    def stop_take(self):
        with self.lock:
            if self.video is not None:
                self.video.close()
                self.times.close()
                self.video = self.times = None

    @property
    def recording(self):
        return self.video is not None

    def run(self):
        while self.running:
            ok, data = self.cap.read()
            now = time.monotonic()
            with self.lock:
                if ok and self.video is not None:
                    self.first = self.first if self.first is not None else now
                    if self.last is not None and now - self.last > 1.5 / FPS:
                        self.late += 1
                    self.last = now
                    self.video.write(data.tobytes())
                    self.times.write(f"{self.written},{now - self.first:.4f}\n")
                    self.written += 1
            with self.ready:
                if not ok:
                    self.running = False
                else:
                    self.frame, self.stamp, self.count = data, now, self.count + 1
                self.ready.notify_all()


def glare_fraction(gray, corners):
    pts = corners.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    return float((gray[max(y0, 0):y1, max(x0, 0):x1] >= GLARE_LEVEL).mean())


def observe(gray, geo, K, previous, dt):
    """Analyse one frame; returns the sample and the board check."""
    check = evaluate(gray, geo, K, Limits(), previous, dt)
    marker = cantos_marcador(gray, geo) is not None
    board = check.corners is not None
    sample = Sample(dt=dt or 0.0, board=board, marker=marker, speed=check.speed, corners=check.corners)
    if board and check.tilt_deg is not None:
        sample.tilt = check.tilt_deg
        sample.distance_m = check.distance_mm / 1000
        sample.side = tilt_side(check.normal)
        sample.glare = glare_fraction(gray, check.corners)
    return sample, check


def run(cap, guide_class, folder, geo, K):
    recorder = Recorder(cap)
    recorder.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    guide = guide_class()
    previous, previous_stamp, count, take = None, None, 0, None
    try:
        while True:
            item = recorder.next_frame(count)
            if item is None:
                print("A câmera parou de enviar quadros.")
                break
            data, stamp, count = item
            frame = cv2.imdecode(data.ravel(), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dt = None if previous_stamp is None else stamp - previous_stamp
            sample, check = observe(gray, geo, K, previous, dt)
            previous, previous_stamp = check.corners, stamp
            if recorder.recording:
                guide.update(sample)

            if check.corners is not None:
                cv2.drawChessboardCorners(frame, geo["padrao"], check.corners, True)
            if recorder.recording:
                state = (f"GRAVANDO {take.name}: {recorder.written} quadros, {recorder.late} atrasados", RED)
                lines = [(guide.title, WHITE), state, *guide.lines(), *guide.common_lines()]
                if guide.done:
                    lines.append(("Meta atingida: aperte espaco para parar", GREEN))
            else:
                tilt = "--" if sample.tilt is None else f"{sample.tilt:.0f}"
                distance = "--" if sample.distance_m is None else f"{sample.distance_m:.2f}"
                lines = [(guide.title, WHITE),
                         (f"Parado. Inclinacao {tilt} graus, distancia {distance} m", WHITE),
                         ("Espaco: gravar   Q: sair", YELLOW)]
                if check.light:
                    lines.append((check.light, YELLOW))
            banner(frame, lines)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if recorder.recording:
                    recorder.stop_take()
                    print(f"{take.name}: {recorder.written} quadros, {recorder.late} atrasados. {guide.summary()}.")
                else:
                    guide = guide_class()
                    take = next_take(folder, folder_scene(guide_class))
                    recorder.start_take(take)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if recorder.recording:
            recorder.stop_take()
            print(f"{take.name}: {recorder.written} quadros, {recorder.late} atrasados. {guide.summary()}.")
        recorder.running = False
        recorder.join(timeout=2)
        cv2.destroyAllWindows()


def folder_scene(guide_class):
    return next(name for name, cls in SCENES.items() if cls is guide_class)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cena", choices=sorted(SCENES))
    ap.add_argument("--device", help="video device (default: first camera found)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--exposure", type=int, default=150, help="exposure_time_absolute, in units of 100 us")
    ap.add_argument("--gain", type=int, default=8)
    ap.add_argument("--white-balance", type=int, default=4000, help="colour temperature (K)")
    ap.add_argument("--focal-px", type=float, default=1420, help="nominal focal length for the live guidance")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    device = a.device or find_camera()
    print(f"Câmera: {device}")
    cap = open_camera(device, raw_jpeg=True)
    try:
        lock_controls(device, a.exposure, a.gain, a.white_balance)
        run(cap, SCENES[a.cena], a.out, carregar_geometria(), nominal_camera(a.focal_px))
    finally:
        cap.release()


if __name__ == "__main__":
    main()
