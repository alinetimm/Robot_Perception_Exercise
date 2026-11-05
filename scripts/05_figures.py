"""Paper figures. Reads data/proc/, results/ and one calibration photo (Figure 1), and
writes figures/ (vector PDF and PNG).

Series differ by marker and line style as well as colour, so every figure survives
greyscale printing. Run: .venv/bin/python scripts/05_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import cantos_marcador, cantos_tabuleiro, carregar_geometria_medida  # noqa: E402
from src.covariance import CHI2_6_HIGH, CHI2_6_LOW, CHI2_6_MEDIAN, jacobian, pose_covariance  # noqa: E402
from src.live import tip_ellipse  # noqa: E402
from src.pose import load_camera, marker_model, marker_poses  # noqa: E402

OUT = ROOT / "figures"
ONE_COLUMN, TWO_COLUMNS = 3.5, 7.16
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# Colours follow the entity across all figures.
MODELS = {
    "fixed": dict(label="Fixed", color="#2a78d6", marker="o", ls="-"),
    "residual": dict(label="Residual", color="#eda100", marker="v", ls=":"),
    "effective": dict(label="Effective", color="#1baf7a", marker="s", ls="--"),
    "speed": dict(label="Speed", color="#eb6834", marker="D", ls="-."),
}
VIDEOS = {
    "E1": dict(color="#2a78d6", marker="o", ls="-"),
    "E2": dict(color="#eb6834", marker="s", ls="--"),
    "E3": dict(color="#1baf7a", marker="^", ls=":"),
}
METHODS = {
    "hold": dict(label="Hold last pose", color="#52514e", marker="o", ls="--"),
    "ring": dict(label="Ring tracking", color="#eb6834", marker="s", ls="-"),
}
SETUP_PHOTO = "data/raw/E0_stills/still_0222.png"
SETUP_MARGIN_PX = 75                 # around the board and the marker
SETUP_ELLIPSE_GAIN = 5.0             # stated in the caption
ROWS = ["E1", "E2", "E3", "E5", "E1_far"]
ROW_LABELS = {"E1": "E1 (fit)", "E2": "E2", "E3": "E3", "E5": "E5", "E1_far": "Far, 1.66 m"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "lines.linewidth": 1.2, "lines.markersize": 4, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def plain_log(axis):
    """Log axis labelled 1, 2, 5, 10 instead of powers of ten."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def setup():
    """A calibration photo with the marker axes and the 3-sigma ellipse of the pipeline."""
    geo = carregar_geometria_medida()
    model = marker_model(geo["objp_marcador"])
    camera = load_camera(ROOT / "calib/camera.yaml")
    speed_model = json.loads((ROOT / "results/nees.json").read_text(encoding="utf-8"))["speed_model"]
    frame = cv2.imread(str(ROOT / SETUP_PHOTO))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker = cantos_marcador(gray, geo).reshape(-1, 2)
    board = cantos_tabuleiro(gray, geo).reshape(-1, 2)
    pose = marker_poses(marker.reshape(-1, 1, 2), model, camera)[0]
    cov = pose_covariance(jacobian(model.points, pose.R, pose.t, camera), speed_model["base_px"])
    rvec = cv2.Rodrigues(pose.R)[0]
    cv2.drawFrameAxes(frame, camera.K, camera.dist, rvec, pose.t.reshape(3, 1), 50.0, 3)
    centre, axes, angle = tip_ellipse(pose.R, pose.t, cov, camera, 100.0, 3.0, SETUP_ELLIPSE_GAIN)
    cv2.ellipse(frame, tuple(np.rint(centre).astype(int)), tuple(np.rint(axes).astype(int)), angle, 0, 360,
                (0, 255, 255), 3, cv2.LINE_AA)
    points = np.vstack([board, marker, centre + axes[0], centre - axes[0]])
    x0, y0 = np.maximum(points.min(0) - SETUP_MARGIN_PX, 0).astype(int)
    x1, y1 = np.minimum(points.max(0) + SETUP_MARGIN_PX, frame.shape[1::-1]).astype(int)
    fig, ax = plt.subplots(figsize=(ONE_COLUMN, ONE_COLUMN * (y1 - y0) / (x1 - x0)), constrained_layout=True)
    ax.imshow(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.25", fc="white", ec=MUTED, lw=0.5)
    ring = (centre[0] + axes[0] * 0.71, centre[1] + axes[0] * 0.71)
    corner = marker[1]
    right = x1 - 0.2 * (x1 - x0)
    for text, target, place in (("Reference chessboard", board.mean(0), (board.mean(0)[0], y0 + 25)),
                                ("Marker", corner, (right, corner[1] - 25)),
                                (r"3$\sigma$ ellipse", ring, (right, y1 - 30))):
        ax.annotate(text, xy=(target[0] - x0, target[1] - y0), xytext=(place[0] - x0, place[1] - y0), bbox=box,
                    ha="center", va="center", arrowprops=dict(arrowstyle="-", color="white", lw=1.0))
    save(fig, "fig1_setup")


def pipeline():
    fig, ax = plt.subplots(figsize=(TWO_COLUMNS, 2.75), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 39)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fill="#ffffff", dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fill, ec=INK,
                                    lw=0.7, ls=(0, (4, 2)) if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.15)

    def arrow(points, dashed=False):
        for a, b, last in zip(points[:-1], points[1:], [False] * (len(points) - 2) + [True]):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if last else "-", mutation_scale=7, lw=0.7, color=INK,
                                         ls=(0, (4, 2)) if dashed else "-", shrinkA=0, shrinkB=0))

    grey, tint = "#efeeea", "#dbe8f8"
    box(0, 15, 11, 10, "Camera\nframe")
    box(0, 33, 100, 5, "Offline, from photos: $K$, distortion, measured 3D board shape, measured marker corners", grey)

    box(16, 25, 18, 5, "Chessboard\n40 corners", dashed=True)
    box(40, 25, 18, 5, "Board pose\nPnP", dashed=True)
    box(64, 25, 15, 5, "Expected\nmarker pose", dashed=True)

    box(16, 14, 18, 7, "ArUco corners\nedge line fit")
    box(40, 14, 18, 7, "Marker pose\nIPPE, two solutions")
    box(64, 14, 15, 7, r"$\Sigma = \sigma^2(v)(J^TJ)^{-1}$", tint)
    box(84, 18, 16, 8, "Error $e$, NEES", tint)

    box(16, 2, 18, 8, "Ring points\nLucas-Kanade,\nhomography")
    box(40, 2, 18, 8, "Tracked corners\nIPPE near last pose")
    box(64, 2, 15, 8, r"$\Sigma_k = \Sigma + kQ$", tint)

    arrow([(11, 20), (13.5, 20), (13.5, 27.5), (16, 27.5)], dashed=True)
    arrow([(11, 20), (13.5, 20), (13.5, 17.5), (16, 17.5)])
    arrow([(13.5, 17.5), (13.5, 6), (16, 6)])
    ax.text(14.2, 12, "marker\nlost", ha="left", va="center", fontsize=8, color=MUTED, linespacing=1.0)
    arrow([(34, 27.5), (40, 27.5)], dashed=True)
    arrow([(58, 27.5), (64, 27.5)], dashed=True)
    arrow([(79, 27.5), (92, 27.5), (92, 26)], dashed=True)
    arrow([(34, 17.5), (40, 17.5)])
    arrow([(58, 17.5), (64, 17.5)])
    arrow([(79, 17.5), (84, 20)])
    arrow([(34, 6), (40, 6)])
    arrow([(58, 6), (64, 6)])
    arrow([(79, 6), (92, 6), (92, 18)])
    ax.text(100, -2.6, "Dashed: reference, used only for evaluation", ha="right", va="bottom", color=MUTED)
    save(fig, "fig2_pipeline")


def nees_table(name):
    return pd.read_parquet(ROOT / f"data/proc/nees/{name}.parquet")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def chi2_lines(ax, horizontal=True):
    draw = ax.axhline if horizontal else ax.axvline
    for value in (CHI2_6_LOW, CHI2_6_HIGH):
        draw(value, color=INK, lw=0.8, ls=(0, (4, 2)))


def error_over_time():
    t = nees_table("E2")
    fig, axes = plt.subplots(2, 1, figsize=(TWO_COLUMNS, 2.9), sharex=True, constrained_layout=True)
    panels = [(axes[0], "e_rot_x_deg", "sd_rot_x_deg", "sd_speed_rot_x_deg", r"$\delta\theta_x$ (deg)"),
              (axes[1], "e_t_z_mm", "sd_t_z_mm", "sd_speed_t_z_mm", r"$\delta t_z$ (mm)")]
    for ax, error, fixed, speed, label in panels:
        ax.fill_between(t.time_s, -2 * t[speed], 2 * t[speed], color=MODELS["speed"]["color"], alpha=0.25, lw=0,
                        label=r"$\pm 2\sigma$, speed model")
        ax.fill_between(t.time_s, -2 * t[fixed], 2 * t[fixed], color=MODELS["fixed"]["color"], alpha=0.55, lw=0,
                        label=r"$\pm 2\sigma$, fixed noise")
        ax.plot(t.time_s, t[error], color=INK, lw=0.7, label="Marker error")
        limit = 1.15 * np.percentile(np.abs(t[error]), 99.5)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel(label)
    axes[0].legend(ncols=3, loc="upper right", bbox_to_anchor=(1, 1.3))
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(t.time_s.min(), t.time_s.max())
    save(fig, "fig3_error_time")


def nees_summary(results):
    t = nees_table("E2")
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMNS, 2.5), width_ratios=[1.5, 1, 1], constrained_layout=True)
    ax = axes[0]
    for key, column in (("fixed", "nees_fixed"), ("speed", "nees_speed")):
        style = MODELS[key]
        ax.plot(t.time_s, t[column], ls="none", marker=style["marker"], ms=1.6, mew=0, color=style["color"],
                label=style["label"])
    chi2_lines(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES, E2")
    ax.set_title("(a) Per frame")
    ax.legend(loc="upper left", ncols=2, markerscale=2.5, handletextpad=0.1)

    y = np.arange(len(ROWS))[::-1]
    for ax, statistic, title, target in ((axes[1], "median", "(b) Median NEES", CHI2_6_MEDIAN),
                                         (axes[2], "anees", "(c) ANEES", 6.0)):
        for offset, (key, style) in zip((0.24, 0.08, -0.08, -0.24), MODELS.items()):
            values = [results["videos"][row][key][statistic] for row in ROWS]
            ax.plot(values, y + offset, ls="none", marker=style["marker"], color=style["color"], ms=3.5,
                    label=style["label"])
        ax.axvline(target, color=INK, lw=0.8, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([ROW_LABELS[r] for r in ROWS])
    axes[2].set_yticklabels([])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower right", ncols=4, handletextpad=0.1, columnspacing=1.0,
               title=r"Noise model in (b) and (c)", title_fontsize=8)
    axes[1].set_xlabel("NEES")
    axes[2].set_xlabel("ANEES")
    save(fig, "fig4_nees")


def by_bins(table, column, edges, statistics, min_frames=30):
    groups = table.groupby(pd.cut(table[column], edges), observed=True)
    rows = []
    for interval, part in groups:
        if len(part) >= min_frames:
            rows.append({"x": interval.mid, **{k: f(part) for k, f in statistics.items()}})
    return pd.DataFrame(rows)


def tilt():
    edges = np.arange(0, 81, 10)
    statistics = {"rot": lambda p: p.err_rot_deg.median(), "nees": lambda p: p.nees_speed.median(),
                  "second": lambda p: 100 * p.second_closer.mean()}
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMNS, 2.2), sharex=True, constrained_layout=True)
    for name, style in VIDEOS.items():
        data = by_bins(nees_table(name), "tilt_deg", edges, statistics)
        for ax, key in zip(axes, ("rot", "nees", "second")):
            ax.plot(data.x, data[key], label=name, **style)
    axes[0].set_ylabel("Median rotation error (deg)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Median NEES, speed model")
    axes[1].set_yscale("log")
    plain_log(axes[1].yaxis)
    axes[1].axhline(CHI2_6_MEDIAN, color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[2].set_ylabel("Wrong IPPE solution (%)")
    axes[2].set_ylim(bottom=0)
    for ax, title in zip(axes, ("(a) Error", "(b) Consistency", "(c) Ambiguity")):
        ax.set_xlabel("Target tilt (deg)")
        ax.set_title(title)
    axes[2].legend(loc="upper right")
    save(fig, "fig5_tilt")


def occlusion():
    tracks = pd.read_parquet(ROOT / "data/proc/occlusion/tracks.parquet")
    tracks = tracks[tracks.set.isin(["E2_test", "E3"])]
    edges = [0, 2, 5, 10, 15, 20, 30, 45, 60]
    statistics = {"rot": lambda p: p.rot_deg.median(), "rot95": lambda p: p.rot_deg.quantile(0.95),
                  "trans": lambda p: p.trans_mm.median(), "trans95": lambda p: p.trans_mm.quantile(0.95),
                  "nees": lambda p: p.nees_growth.median(), "lost": lambda p: 100 * p.lost.mean()}
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COLUMNS, 2.2), sharex=True, constrained_layout=True)
    for method, style in METHODS.items():
        data = by_bins(tracks[tracks.method == method], "k", edges, statistics, min_frames=20)
        for ax, key in zip(axes, ("rot", "trans", "nees")):
            ax.plot(data.x, data[key], **style)
    data = by_bins(tracks[tracks.method == "ring"], "k", edges, statistics, min_frames=20)
    axes[3].plot(data.x, data.lost, **METHODS["ring"])
    p95 = dict(color=METHODS["ring"]["color"], ls=":", lw=0.9, marker=None, label="Ring tracking, 95th percentile")
    axes[0].plot(data.x, data.rot95, **p95)
    axes[1].plot(data.x, data.trans95, **p95)
    for ax in axes[:3]:
        ax.set_yscale("log")
        plain_log(ax.yaxis)
    axes[0].set_ylabel("Rotation error (deg)")
    axes[1].set_ylabel("Translation error (mm)")
    axes[2].set_ylabel(r"Median NEES with $\Sigma_k$")
    axes[2].axhline(CHI2_6_MEDIAN, color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[3].set_ylabel("Ring track lost (%)")
    axes[3].set_ylim(0, 100)
    for ax, title in zip(axes, ("(a) Rotation", "(b) Translation", "(c) Consistency", "(d) Survival")):
        ax.set_xlabel("Frames since loss")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    save(fig, "fig6_occlusion")


def speed():
    table = pd.concat([nees_table(n) for n in ("E2", "E3", "E5")], ignore_index=True)
    edges = [0, 0.3, 0.6, 1, 2, 3, 5, 10, 20, 60]
    statistics = {k: (lambda p, c=f"nees_{k}": p[c].median()) for k in ("fixed", "effective", "speed")}
    fig, ax = plt.subplots(figsize=(ONE_COLUMN, 2.4), constrained_layout=True)
    data = by_bins(table, "speed", edges, statistics)
    for key in ("fixed", "effective", "speed"):
        style = MODELS[key]
        ax.plot(data.x, data[key], label=style["label"], color=style["color"], marker=style["marker"], ls=style["ls"])
    ax.axhline(CHI2_6_MEDIAN, color=INK, lw=0.8, ls=(0, (4, 2)), label=r"$\chi^2_6$ median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax.xaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Marker speed (px/frame)")
    ax.set_ylabel("Median NEES, E2, E3, E5")
    ax.legend(loc="upper left", ncols=2, columnspacing=0.8)
    save(fig, "fig7_speed")


def main():
    results = json.loads((ROOT / "results/nees.json").read_text(encoding="utf-8"))
    setup()
    pipeline()
    error_over_time()
    nees_summary(results)
    tilt()
    occlusion()
    speed()


if __name__ == "__main__":
    main()
