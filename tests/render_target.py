"""Render the printed target at a known pose, for tests that need real pixels."""
from pathlib import Path

import cv2
import numpy as np

from alvo.referencia import cantos_tabuleiro, carregar_geometria

PREVIEW = Path(__file__).resolve().parents[1] / "alvo" / "alvo_A4_preview.png"
SIZE = (1280, 960)
BOARD_CENTER = np.array([70.0, 40.0, 0.0])


class TargetRenderer:
    def __init__(self, K):
        self.K = np.asarray(K, float)
        self.geo = carregar_geometria()
        self.page = cv2.imread(str(PREVIEW), cv2.IMREAD_GRAYSCALE)
        corners = cantos_tabuleiro(self.page, self.geo)
        self.board_to_page = cv2.findHomography(self.geo["objp_tabuleiro"][:, :2], corners.reshape(-1, 2))[0]

    def pose(self, tilt_deg=0.0, tilt_axis_deg=0.0, center_px=(640, 480), distance_mm=500.0):
        """Rotation and translation that place the board centre on a pixel."""
        axis = np.radians(tilt_axis_deg)
        rvec = np.radians(tilt_deg) * np.array([np.cos(axis), np.sin(axis), 0.0])
        R = cv2.Rodrigues(rvec)[0]
        ray = np.linalg.inv(self.K) @ np.array([*center_px, 1.0])
        return R, distance_mm * ray - R @ BOARD_CENTER

    def render(self, R, t):
        board_to_image = self.K @ np.column_stack([R[:, 0], R[:, 1], t])
        M = board_to_image @ np.linalg.inv(self.board_to_page)
        image = cv2.warpPerspective(self.page, M, SIZE, flags=cv2.INTER_AREA, borderValue=110)
        return cv2.GaussianBlur(image, (0, 0), 0.6)


def distortion_map(K, dist):
    """Remap that turns an ideal image into one with lens distortion `dist`."""
    xs, ys = np.meshgrid(np.arange(SIZE[0], dtype=np.float64), np.arange(SIZE[1], dtype=np.float64))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, np.asarray(K, float), np.asarray(dist, float), P=np.asarray(K, float))
    ideal = ideal.reshape(SIZE[1], SIZE[0], 2).astype(np.float32)
    return ideal[..., 0], ideal[..., 1]
