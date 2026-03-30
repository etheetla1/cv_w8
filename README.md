# Stereo Classroom Object Localization

Compute 2D floor-plane locations of tables and chairs in a classroom using a simple stereo camera setup, then visualize them on an X-Y bird's-eye plot (tables in red, chairs in blue).

## Output

![Classroom Floor Plan](classroom_floorplan.png)

- **Red squares** = Tables
- **Blue circles** = Chairs
- Axes are in relative units (no absolute calibration available)

## How It Works

### Pipeline Overview

```
Load stereo pair (C1, C5)
        |
        ├──> SIFT feature matching ──> Essential Matrix ──> Camera Pose
        |
        └──> YOLOv8 object detection (both images)
                    |
                    v
        Match detections across views (Hungarian algorithm)
                    |
                    v
        Compute depth via disparity: depth = f / |disparity|
                    |
                    v
        Project to X-Y floor plane ──> Plot & save
```

### Step-by-step

1. **Load stereo pair** -- `C1.jpeg` (left) and `C5.jpeg` (right), both 4032x3024 landscape images taken from slightly different positions in the classroom.

2. **Estimate camera geometry** -- SIFT extracts ~8000 keypoints per image. FLANN matcher with Lowe's ratio test finds ~3000 good correspondences. The Essential Matrix is computed via RANSAC, and camera pose (R, t) is recovered with `cv2.recoverPose`.

3. **Detect objects** -- YOLOv8-medium (`yolov8m.pt`) runs on both full-resolution images at a 0.15 confidence threshold. It filters for COCO classes: `chair` (56), `dining table` (60), and `bench` (13, mapped to table).

4. **Match detections across views** -- The Hungarian algorithm finds optimal 1-to-1 matches between left and right detections. Constraints: same class, vertical proximity (epipolar constraint, <100px), and bounding box size similarity.

5. **Compute depth** -- Since the baseline between C1 and C5 is very small (handheld shift), full triangulation is unreliable. The script automatically falls back to **disparity-based depth**: `depth = focal_length / |horizontal_disparity|`, which is robust for small baselines. Objects with disparity < 2px or depth < 15 (relative units) are filtered as unreliable.

6. **Project to floor plane** -- X-world coordinate is computed as `(x_pixel - cx) * depth / focal_length`. Outliers are removed via Median Absolute Deviation (MAD) filtering.

7. **Plot** -- `matplotlib` renders the 2D bird's-eye floor plan with tables as red squares and chairs as blue circles.

### Key Assumptions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Focal length | `0.85 * image_width` px | Typical smartphone ~26mm equivalent |
| Principal point | Image center | Standard assumption |
| Baseline | Unknown | Coordinates are in relative units |
| Min disparity | 2 px | Below this, depth is unreliable |

## Input Images

Five classroom photos in `classroom_images/`:

| Image | Resolution | Orientation | Role |
|-------|-----------|-------------|------|
| C1.jpeg | 4032x3024 | Landscape | Left stereo view |
| C2.jpeg | 3024x4032 | Portrait | Unused |
| C3.jpeg | 3024x4032 | Portrait | Unused |
| C4.jpeg | 3024x4032 | Portrait | Unused |
| C5.jpeg | 4032x3024 | Landscape | Right stereo view |

C1 and C5 were selected as the stereo pair because they share the same landscape orientation and have a small horizontal baseline.

## How to Run

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python3 stereo_classroom.py
```

The script prints step-by-step progress and saves `classroom_floorplan.png`.

## Dependencies

Listed in `requirements.txt`:

- `opencv-python` / `opencv-contrib-python` -- Feature matching, stereo geometry
- `ultralytics` -- YOLOv8 object detection
- `matplotlib` -- 2D plotting
- `numpy` -- Numerical computation
- `scipy` -- Hungarian algorithm for detection matching

## Project Structure

```
cv_w8/
├── classroom_images/
│   ├── C1.jpeg          # Left stereo image
│   ├── C2.jpeg
│   ├── C3.jpeg
│   ├── C4.jpeg
│   └── C5.jpeg          # Right stereo image
├── stereo_classroom.py  # Main pipeline script
├── requirements.txt     # Python dependencies
├── classroom_floorplan.png  # Output 2D floor plan
└── README.md
```

## Results Summary

| Metric | Value |
|--------|-------|
| SIFT keypoints | ~6300 / ~7000 per image |
| Good feature matches | ~3055 |
| Essential matrix inliers | ~2440 |
| Left detections | 1 table, 16 chairs |
| Right detections | 1 table, 20 chairs |
| Cross-view matched pairs | 17 (1 table, 16 chairs) |
| Final plotted objects | 8 (1 table, 7 chairs) |
| Depth method | Disparity-based (small baseline) |

## Limitations

- **Table detection**: YOLO's COCO "dining table" class does not match small classroom desks well, especially when occluded by students and laptops. Only 1 table was reliably detected per image.
- **Small baseline**: The two images were taken with a very small horizontal shift (handheld), so full stereo triangulation via pose recovery is noisy. The disparity-based fallback provides a more stable result.
- **Relative coordinates**: Without a known baseline distance, all positions are in relative (not metric) units. The spatial arrangement is correct but not to scale.
