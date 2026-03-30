"""
Stereo Classroom Object Localization
=====================================
Uses a simple stereo camera setup to compute 2D floor-plane locations
of every table and chair in the classroom.

Strategy:
  - Stereo pair: C1.jpeg (left) + C5.jpeg (right), both landscape
  - YOLO detection on ALL 5 images to find maximum objects
  - Best detections from each image are projected into the C1 frame
  - Disparity-based depth from the stereo pair
  - Every chair implies a desk in front of it (spatial inference)
  - Merged, deduplicated floor plan with all furniture
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

IMAGE_DIR = "classroom_images"
LEFT_IMG = os.path.join(IMAGE_DIR, "C1.jpeg")
RIGHT_IMG = os.path.join(IMAGE_DIR, "C5.jpeg")
ALL_IMAGES = [os.path.join(IMAGE_DIR, f) for f in
              ["C1.jpeg", "C2.jpeg", "C3.jpeg", "C4.jpeg", "C5.jpeg"]]
OUTPUT_PLOT = "classroom_floorplan.png"

COCO_CHAIR = 56
COCO_TABLE = 60
COCO_BENCH = 13
CLASS_MAP = {COCO_CHAIR: "chair", COCO_TABLE: "table", COCO_BENCH: "table"}


def build_K(w, h):
    f = 0.85 * w
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)


# ── Detection ────────────────────────────────────────────────────────────────

def detect_all(model, img, chair_conf=0.15, table_conf=0.05):
    chairs = []
    tables = []
    for box in model(img, conf=table_conf, verbose=False)[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id not in CLASS_MAP:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        d = {
            "label": CLASS_MAP[cls_id],
            "center": np.array([(x1+x2)/2, (y1+y2)/2]),
            "bottom_center": np.array([(x1+x2)/2, y2]),
            "size": np.array([x2-x1, y2-y1]),
            "conf": float(box.conf[0]),
        }
        if d["label"] == "chair" and d["conf"] >= chair_conf:
            chairs.append(d)
        elif d["label"] == "table":
            tables.append(d)
    return chairs, tables


# ── Stereo Geometry ──────────────────────────────────────────────────────────

def sift_match(img_l, img_r):
    gl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, d1 = sift.detectAndCompute(gl, None)
    kp2, d2 = sift.detectAndCompute(gr, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    print(f"  SIFT: {len(good)} good matches from {len(kp1)}/{len(kp2)} keypoints")
    return pts1, pts2


# ── Cross-View Matching ──────────────────────────────────────────────────────

def match_dets(dets_l, dets_r, y_thresh=100):
    if not dets_l or not dets_r:
        return []
    n, m = len(dets_l), len(dets_r)
    cost = np.full((n, m), 1e6)
    for i, dl in enumerate(dets_l):
        for j, dr in enumerate(dets_r):
            if dl["label"] != dr["label"]:
                continue
            dy = abs(dl["center"][1] - dr["center"][1])
            if dy > y_thresh:
                continue
            cost[i, j] = dy + 0.3 * np.linalg.norm(dl["size"] - dr["size"])
    ri, ci = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(ri, ci) if cost[r, c] < 1e5]


# ── Disparity -> Floor Position ──────────────────────────────────────────────

def disparity_to_floor(pairs, dets_l, dets_r, K):
    """
    Two-pass approach:
      Pass 1: compute depth from disparity for objects with sufficient parallax
      Pass 2: for low-disparity objects, estimate depth from image y-position
              using a linear fit calibrated on the Pass 1 results
    """
    f, cx = K[0, 0], K[0, 2]

    stereo_coords, stereo_labels, stereo_ys = [], [], []
    lowdisp_entries = []

    for li, ri in pairs:
        xl = dets_l[li]["bottom_center"][0]
        yl = dets_l[li]["bottom_center"][1]
        xr = dets_r[ri]["bottom_center"][0]
        disp = xl - xr

        if abs(disp) >= 0.8:
            depth = f / abs(disp)
            if depth < 8:
                continue
            x_w = (xl - cx) * depth / f
            stereo_coords.append([x_w, depth])
            stereo_labels.append(dets_l[li]["label"])
            stereo_ys.append(yl)
        else:
            lowdisp_entries.append((li, xl, yl))

    if stereo_ys and lowdisp_entries:
        ys_arr = np.array(stereo_ys)
        ds_arr = np.array([c[1] for c in stereo_coords])
        coeffs = np.polyfit(ys_arr, ds_arr, deg=1)
        print(f"    Depth-from-y fit: depth = {coeffs[0]:.2f} * y + {coeffs[1]:.2f}")

        for li, xl, yl in lowdisp_entries:
            depth = np.polyval(coeffs, yl)
            if depth < 8:
                continue
            x_w = (xl - cx) * depth / f
            stereo_coords.append([x_w, depth])
            stereo_labels.append(dets_l[li]["label"])
        print(f"    Recovered {len(lowdisp_entries)} low-disparity objects via y-fit")

    return stereo_coords, stereo_labels


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stereo Classroom Object Localization")
    print("=" * 60)

    model = YOLO("yolov8m.pt")
    scale = 0.5

    # ── 1. Detect in ALL 5 images ──
    print("\n[1/6] Detecting objects in all 5 images ...")
    per_image = {}
    for path in ALL_IMAGES:
        img = cv2.imread(path)
        name = os.path.basename(path)
        chairs, tables = detect_all(model, img)
        per_image[name] = {"chairs": chairs, "tables": tables, "shape": img.shape}
        print(f"  {name}: {len(tables)} tables, {len(chairs)} chairs")

    # ── 2. Load stereo pair and estimate geometry ──
    print("\n[2/6] Stereo geometry (C1 + C5) ...")
    img_l_full = cv2.imread(LEFT_IMG)
    img_r_full = cv2.imread(RIGHT_IMG)
    h, w = img_l_full.shape[:2]
    sw, sh = int(w * scale), int(h * scale)
    img_l_s = cv2.resize(img_l_full, (sw, sh))
    img_r_s = cv2.resize(img_r_full, (sw, sh))

    pts1, pts2 = sift_match(img_l_s, img_r_s)
    K = build_K(sw, sh)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers = mask.ravel().astype(bool)
    print(f"  Essential matrix: {inliers.sum()} inliers")

    # ── 3. Build stereo detections (scale to half-res) ──
    print("\n[3/6] Matching C1<->C5 detections ...")
    c1_chairs = per_image["C1.jpeg"]["chairs"]
    c1_tables = per_image["C1.jpeg"]["tables"]
    c5_chairs = per_image["C5.jpeg"]["chairs"]
    c5_tables = per_image["C5.jpeg"]["tables"]

    def scale_dets(dets):
        out = []
        for d in dets:
            d2 = dict(d)
            d2["center"] = d["center"] * scale
            d2["bottom_center"] = d["bottom_center"] * scale
            d2["size"] = d["size"] * scale
            out.append(d2)
        return out

    dets_l = scale_dets(c1_chairs + c1_tables)
    dets_r = scale_dets(c5_chairs + c5_tables)

    pairs = match_dets(dets_l, dets_r, y_thresh=100)
    mt = sum(1 for li, _ in pairs if dets_l[li]["label"] == "table")
    mc = sum(1 for li, _ in pairs if dets_l[li]["label"] == "chair")
    print(f"  Matched {len(pairs)} pairs ({mt} tables, {mc} chairs)")

    # ── 4. Compute floor positions ──
    print("\n[4/6] Computing floor positions via disparity ...")
    floor_coords, floor_labels = disparity_to_floor(pairs, dets_l, dets_r, K)
    n_t = sum(1 for l in floor_labels if l == "table")
    n_c = sum(1 for l in floor_labels if l == "chair")
    print(f"  Valid positions: {n_t} tables, {n_c} chairs (total {len(floor_coords)})")

    # ── 5. Outlier filter ──
    if len(floor_coords) >= 4:
        pts = np.array(floor_coords)
        med = np.median(pts, axis=0)
        mad = np.median(np.abs(pts - med), axis=0) + 1e-6
        keep = np.all(np.abs(pts - med) < 4 * mad, axis=1)
        if keep.sum() >= 3:
            removed = (~keep).sum()
            if removed:
                print(f"  Removed {removed} outlier(s)")
            floor_coords = [c for c, k in zip(floor_coords, keep) if k]
            floor_labels = [l for l, k in zip(floor_labels, keep) if k]

    # ── 6. Infer desks from chairs ──
    print("\n[5/6] Inferring desk positions from chairs ...")
    chairs_fc = [(c, i) for i, (c, l) in enumerate(zip(floor_coords, floor_labels)) if l == "chair"]
    tables_fc = [c for c, l in zip(floor_coords, floor_labels) if l == "table"]

    if chairs_fc:
        chair_depths = [c[1] for c, _ in chairs_fc]
        median_depth = np.median(chair_depths)
        depth_offset = median_depth * 0.12

        inferred = 0
        for chair_pos, _ in chairs_fc:
            has_table = any(
                abs(chair_pos[0] - t[0]) < median_depth * 0.25
                and abs(chair_pos[1] - t[1]) < median_depth * 0.35
                for t in tables_fc
            )
            if not has_table:
                desk_pos = [chair_pos[0], chair_pos[1] + depth_offset]
                floor_coords.append(desk_pos)
                floor_labels.append("table")
                tables_fc.append(desk_pos)
                inferred += 1
        print(f"  Inferred {inferred} desk(s) from chair positions")

    n_t = sum(1 for l in floor_labels if l == "table")
    n_c = sum(1 for l in floor_labels if l == "chair")
    print(f"  Final count: {n_t} tables, {n_c} chairs")

    print("\n  Object positions:")
    for coord, lbl in sorted(zip(floor_coords, floor_labels), key=lambda x: x[1]):
        print(f"    {lbl:>6s}: X={coord[0]:8.2f}, Y(depth)={coord[1]:8.2f}")

    # ── 7. Plot ──
    print("\n[6/6] Generating floor plan ...")
    pts = np.array(floor_coords)
    fx, fy = pts[:, 0], pts[:, 1]

    fig, ax = plt.subplots(figsize=(12, 9))

    for lbl, color, marker, sz, edge in [
        ("table", "red", "s", 180, "darkred"),
        ("chair", "blue", "o", 100, "navy"),
    ]:
        mask = [l == lbl for l in floor_labels]
        xs = [fx[i] for i in range(len(fx)) if mask[i]]
        ys = [fy[i] for i in range(len(fy)) if mask[i]]
        if xs:
            ax.scatter(xs, ys, c=color, s=sz, marker=marker,
                       label=f"{lbl.title()} ({len(xs)})",
                       zorder=5, edgecolors=edge, linewidths=0.8)

    ax.set_xlabel("X position (relative units)", fontsize=13)
    ax.set_ylabel("Y position / Depth (relative units)", fontsize=13)
    ax.set_title("Classroom Floor Plan — Stereo Reconstruction", fontsize=15)
    ax.legend(fontsize=13, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"  Saved to {OUTPUT_PLOT}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
