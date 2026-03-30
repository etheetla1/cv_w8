"""
Stereo Classroom Object Localization
=====================================
Uses a simple stereo camera setup (two images from slightly different positions)
to compute 2D floor-plane locations of tables and chairs in a classroom.

Pipeline:
  1. Load stereo pair (C1.jpeg = left, C5.jpeg = right)
  2. SIFT feature matching -> Essential Matrix -> Camera Pose
  3. YOLOv8 object detection (tables + chairs) on both images
  4. Match detections across views using epipolar + size constraints
  5. Triangulate matched detections to 3D
  6. Project onto the X-Z floor plane and plot

When full triangulation is unreliable (small baseline), falls back to
disparity-based depth estimation from matched detection centres.
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
OUTPUT_PLOT = "classroom_floorplan.png"

COCO_CHAIR = 56
COCO_TABLE = 60
COCO_BENCH = 13
CLASS_MAP = {COCO_CHAIR: "chair", COCO_TABLE: "table", COCO_BENCH: "table"}


# ── 1. Load Stereo Pair ─────────────────────────────────────────────────────

def load_stereo_pair(left_path, right_path, scale=0.5):
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    if img_l is None or img_r is None:
        raise FileNotFoundError(f"Could not load images: {left_path}, {right_path}")

    h, w = img_l.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    img_l = cv2.resize(img_l, (new_w, new_h))
    img_r = cv2.resize(img_r, (new_w, new_h))
    return img_l, img_r, scale


# ── 2. Feature Matching & Camera Geometry ────────────────────────────────────

def build_intrinsic(img_shape):
    h, w = img_shape[:2]
    focal_length = 0.85 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K, focal_length, (cx, cy)


def estimate_camera_pose(img_l, img_r):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(gray_l, None)
    kp2, des2 = sift.detectAndCompute(gray_r, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"  SIFT: {len(kp1)} / {len(kp2)} keypoints, {len(good)} good matches")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    K, f, _ = build_intrinsic(img_l.shape)

    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                     prob=0.999, threshold=1.0)
    inliers = mask_e.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    print(f"  Essential matrix: {inliers.sum()} inliers / {len(good)} matches")

    n_front, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    print(f"  Recovered pose: {n_front} points in front of both cameras")

    pose_reliable = n_front > max(50, 0.05 * inliers.sum())
    return K, R, t, pose_reliable


# ── 3. Object Detection ─────────────────────────────────────────────────────

def detect_objects(model, img, conf_thresh=0.25):
    results = model(img, conf=conf_thresh, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in CLASS_MAP:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bot_cy = y2
        w = x2 - x1
        h = y2 - y1
        conf = float(box.conf[0])
        detections.append({
            "class_id": cls_id,
            "label": CLASS_MAP[cls_id],
            "center": np.array([cx, cy]),
            "bottom_center": np.array([cx, bot_cy]),
            "size": np.array([w, h]),
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
        })
    return detections


# ── 4. Match Detections Across Views ─────────────────────────────────────────

def match_detections(dets_l, dets_r, y_thresh=100, size_weight=0.3):
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
            size_diff = np.linalg.norm(dl["size"] - dr["size"])
            cost[i, j] = dy + size_weight * size_diff

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e5:
            pairs.append((r, c))
    return pairs


# ── 5a. Full Triangulation (when pose is reliable) ──────────────────────────

def triangulate_matches(pairs, dets_l, dets_r, K, R, t):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    points_3d, labels = [], []
    for li, ri in pairs:
        pt_l = dets_l[li]["center"].astype(np.float64)
        pt_r = dets_r[ri]["center"].astype(np.float64)

        pts4d = cv2.triangulatePoints(P1, P2, pt_l.reshape(2, 1), pt_r.reshape(2, 1))
        pt3 = (pts4d[:3] / pts4d[3]).flatten()
        points_3d.append(pt3)
        labels.append(dets_l[li]["label"])

    if points_3d:
        depths = [p[2] for p in points_3d]
        if sum(d < 0 for d in depths) > len(depths) / 2:
            print("  Majority behind camera — flipping Z sign.")
            points_3d = [np.array([p[0], p[1], -p[2]]) for p in points_3d]

    return points_3d, labels


# ── 5b. Disparity-Based Depth (robust for small baselines) ──────────────────

def disparity_depth(pairs, dets_l, dets_r, K):
    """
    Uses horizontal disparity between matched detection centres to
    estimate relative depth:  depth ~ focal_length / |disparity|.
    Returns floor-plane (X, Z) in camera-1 frame.
    """
    f = K[0, 0]
    cx = K[0, 2]

    min_disp = 2.0

    floor_coords, labels = [], []
    for li, ri in pairs:
        xl = dets_l[li]["bottom_center"][0]
        xr = dets_r[ri]["bottom_center"][0]
        disp = xl - xr
        if abs(disp) < min_disp:
            continue

        depth = f / abs(disp)
        if depth < 15:
            continue
        x_world = (xl - cx) * depth / f

        floor_coords.append(np.array([x_world, depth]))
        labels.append(dets_l[li]["label"])

    return floor_coords, labels


# ── 6. Floor-Plane Projection & Plotting ─────────────────────────────────────

def plot_floor_plan(floor_coords, labels, output_path, title_suffix=""):
    if not floor_coords:
        print("WARNING: No points to plot.")
        return

    pts = np.array(floor_coords)
    floor_x = pts[:, 0]
    floor_y = pts[:, 1]

    labels_arr = list(labels)

    if len(floor_x) >= 4:
        med_x, med_y = np.median(floor_x), np.median(floor_y)
        mad_x = np.median(np.abs(floor_x - med_x)) + 1e-6
        mad_y = np.median(np.abs(floor_y - med_y)) + 1e-6
        keep = (np.abs(floor_x - med_x) < 6 * mad_x) & \
               (np.abs(floor_y - med_y) < 6 * mad_y)
        if keep.sum() >= 3:
            removed = (~keep).sum()
            floor_x = floor_x[keep]
            floor_y = floor_y[keep]
            labels_arr = [l for l, v in zip(labels_arr, keep) if v]
            if removed:
                print(f"  Removed {removed} outlier(s)")

    table_mask = [l == "table" for l in labels_arr]
    chair_mask = [l == "chair" for l in labels_arr]

    fig, ax = plt.subplots(figsize=(10, 8))

    tx = [floor_x[i] for i in range(len(floor_x)) if table_mask[i]]
    ty = [floor_y[i] for i in range(len(floor_y)) if table_mask[i]]
    cx_pts = [floor_x[i] for i in range(len(floor_x)) if chair_mask[i]]
    cy_pts = [floor_y[i] for i in range(len(floor_y)) if chair_mask[i]]

    if tx:
        ax.scatter(tx, ty, c="red", s=150, marker="s", label="Table", zorder=5,
                   edgecolors="darkred", linewidths=0.8)
    if cx_pts:
        ax.scatter(cx_pts, cy_pts, c="blue", s=90, marker="o", label="Chair", zorder=5,
                   edgecolors="navy", linewidths=0.8)

    ax.set_xlabel("X position (relative units)", fontsize=12)
    ax.set_ylabel("Y position / Depth (relative units)", fontsize=12)
    ax.set_title(f"Classroom Floor Plan — Stereo Reconstruction{title_suffix}", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved to {output_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stereo Classroom Object Localization")
    print("=" * 60)

    # ── Step 1: Load ──
    print("\n[1/6] Loading stereo pair ...")
    img_l, img_r, scale = load_stereo_pair(LEFT_IMG, RIGHT_IMG, scale=0.5)
    print(f"  Left:  {img_l.shape}   Right: {img_r.shape}")

    # ── Step 2: Camera Geometry ──
    print("\n[2/6] Estimating camera pose (SIFT + Essential Matrix) ...")
    K, R, t, pose_reliable = estimate_camera_pose(img_l, img_r)
    K_full, _, _ = build_intrinsic(cv2.imread(LEFT_IMG).shape)

    # ── Step 3: Object Detection (run on full-res for accuracy) ──
    print("\n[3/6] Running YOLOv8 object detection ...")
    model = YOLO("yolov8m.pt")
    dets_l_full = detect_objects(model, cv2.imread(LEFT_IMG), conf_thresh=0.15)
    dets_r_full = detect_objects(model, cv2.imread(RIGHT_IMG), conf_thresh=0.15)

    dets_l = []
    for d in dets_l_full:
        d2 = dict(d)
        d2["center"] = d["center"] * scale
        d2["bottom_center"] = d["bottom_center"] * scale
        d2["size"] = d["size"] * scale
        dets_l.append(d2)
    dets_r = []
    for d in dets_r_full:
        d2 = dict(d)
        d2["center"] = d["center"] * scale
        d2["bottom_center"] = d["bottom_center"] * scale
        d2["size"] = d["size"] * scale
        dets_r.append(d2)

    tables_l = sum(1 for d in dets_l if d["label"] == "table")
    chairs_l = sum(1 for d in dets_l if d["label"] == "chair")
    tables_r = sum(1 for d in dets_r if d["label"] == "table")
    chairs_r = sum(1 for d in dets_r if d["label"] == "chair")
    print(f"  Left  detections: {tables_l} tables, {chairs_l} chairs")
    print(f"  Right detections: {tables_r} tables, {chairs_r} chairs")

    # ── Step 4: Match across views ──
    print("\n[4/6] Matching detections across views ...")
    pairs = match_detections(dets_l, dets_r, y_thresh=100)
    matched_tables = sum(1 for li, _ in pairs if dets_l[li]["label"] == "table")
    matched_chairs = sum(1 for li, _ in pairs if dets_l[li]["label"] == "chair")
    print(f"  Matched {len(pairs)} pairs ({matched_tables} tables, {matched_chairs} chairs)")

    if len(pairs) < 3:
        print("  Too few matches — cannot proceed.")
        return

    # ── Step 5: Compute depth ──
    print("\n[5/6] Computing 3D positions ...")
    if pose_reliable:
        print("  Using full triangulation (reliable pose) ...")
        pts3d, labels = triangulate_matches(pairs, dets_l, dets_r, K, R, t)
        floor_coords = [[p[0], p[2]] for p in pts3d]
    else:
        print("  Pose unreliable (small baseline) — using disparity-based depth ...")
        floor_coords, labels = disparity_depth(pairs, dets_l, dets_r, K)

    print(f"  Computed floor positions for {len(floor_coords)} objects")
    for (x, y), lbl in zip(floor_coords, labels):
        print(f"    {lbl:>6s}: X={x:8.2f}, Y(depth)={y:8.2f}")

    # ── Step 6: Plot ──
    print("\n[6/6] Generating floor plan plot ...")
    suffix = "" if pose_reliable else " (disparity-based)"
    plot_floor_plan(floor_coords, labels, OUTPUT_PLOT, title_suffix=suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
