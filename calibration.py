import argparse, glob, json, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--square", type=float, required=True)
    ap.add_argument("--simple", action="store_true")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()

    for d in ("corners", "undistorted"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    paths = sorted(p for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                   for p in glob.glob(os.path.join(a.images, f"*.{ext}")))
    if not paths:
        raise SystemExit(f"No images found in {a.images}")

    pattern = (a.cols, a.rows)
    objp = np.zeros((a.rows * a.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square

    obj_pts, img_pts, used, size = [], [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for p in paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"[skip] {os.path.basename(p)}: different resolution {gray.shape[::-1]}")
            continue
        ok, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"[skip] {os.path.basename(p)}: board not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(p)
        vis = cv2.drawChessboardCorners(img.copy(), pattern, corners, ok)
        cv2.imwrite(os.path.join(a.out, "corners", os.path.basename(p)), vis)

    print(f"\nBoard detected in {len(used)} of {len(paths)} images")
    if len(used) < 3:
        raise SystemExit("Too few valid images (aim for 15 to 20).")

    flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST if a.simple else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)

    per_img = []
    for op, ip, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, r, t, K, dist)
        diff = ip.reshape(-1, 2) - proj.reshape(-1, 2)
        per_img.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))

    np.set_printoptions(precision=4, suppress=True)
    print(f"\nImage size: {size[0]} x {size[1]}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print("K =\n", K)
    print("dist (k1, k2, p1, p2, k3) =", dist.ravel())
    print(f"fx/fy = {K[0,0]/K[1,1]:.4f}   principal point offset from center: "
          f"({K[0,2]-size[0]/2:.1f}, {K[1,2]-size[1]/2:.1f}) px")
    worst = np.argsort(per_img)[::-1][:3]
    print("Worst images:", [(os.path.basename(used[i]), round(per_img[i], 3)) for i in worst])

    with open(os.path.join(a.out, "params.json"), "w") as f:
        json.dump({
            "image_size": size, "model": "k1,k2 only" if a.simple else "k1,k2,p1,p2,k3",
            "rms_px": rms, "K": K.tolist(), "dist": dist.ravel().tolist(),
            "per_image_error_px": dict(zip(map(os.path.basename, used), per_img)),
            "extrinsics": [{"image": os.path.basename(p), "rvec": r.ravel().tolist(),
                            "tvec_mm": t.ravel().tolist()} for p, r, t in zip(used, rvecs, tvecs)],
        }, f, indent=2)


    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)
    for p in used:
        img = cv2.imread(p)
        und = cv2.undistort(img, K, dist, None, newK)
        both = np.hstack([img, und])
        cv2.imwrite(os.path.join(a.out, "undistorted", os.path.basename(p)), both)

    # per-image error chart
    names = [os.path.basename(p) for p in used]
    plt.figure(figsize=(max(6, len(names) * 0.45), 4))
    plt.bar(range(len(names)), per_img, color="#E8622C")
    plt.axhline(rms, color="#3B1E2A", ls="--", label=f"overall RMS = {rms:.3f} px")
    plt.xticks(range(len(names)), names, rotation=70, fontsize=7)
    plt.ylabel("reprojection error (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "per_image_error.png"), dpi=150)

    thumbs = [cv2.resize(cv2.imread(p), (240, int(240 * size[1] / size[0]))) for p in used]
    cols = 5
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(os.path.join(a.out, "mosaic_inputs.jpg"), np.vstack(rows))

    print(f"\nResults saved to {a.out}/")


if __name__ == "__main__":
    main()