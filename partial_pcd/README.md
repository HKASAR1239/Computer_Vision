## REALSENSE POINTCLOUD OBJECT ISOLATION

> **Goal**: Capture a colored point cloud with an Intel RealSense, isolate it automatically (or manually) from its environment, and save it cleanly as a `.ply` file, all without complex command-line operations.

---

## 🗂️ Repository Contents

| File                          | Purpose                                                                                                   |
|-------------------------------|-----------------------------------------------------------------------------------------------------------|
| **`v2.py`** | Tkinter interface + live preview: a **"Capture point-cloud"** button triggers capture, isolation, coloring (red), and saving as `.ply`. |
| **`isoler_ply.py`**           | CLI with *all* parameters exposed (`--voxel`, `--eps`, `--min_pts`, etc.) to fine-tune the isolation algorithm. |
| `README.md`                   | (this file) explanations, installation, and examples.                                                     |

---

## ✨ Key Features

* **Real-time color preview** (Tkinter + PIL)
* **One-click capture** with automatic isolation:
  * Down-sampling
  * Outlier removal
  * Plane detection/removal (RANSAC)
  * Density clustering (DBSCAN) → largest cluster = object
* **Open3D display**: black background, red points (adjustable size)
* **Timestamped save**: `isolated_object_YYYYMMDD_HHMMSS.ply`
* **Independent post-processing** for any `.ply` file

---

## 🔧 Prerequisites

| Software                                 | Recommended Version |
|------------------------------------------|---------------------|
| Python                                   | 3.8 – 3.11          |
| Intel RealSense SDK 2.0                  | Latest stable       |
| Open3D                                   | ≥ 0.17              |
| PyPI: `pyrealsense2`, `numpy`, `Pillow`  |                     |

```bash
pip install pyrealsense2 open3d numpy pillow
```

> On Ubuntu, first install the RealSense driver:
> `sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev`

---

## 🚀 Quick Start

```bash
# 1) Clone the main CV project
cd partial_pcd

# 2) Launch the live interface
python v2.py
```

* Connect the camera to a **USB 3.0 port**.
* Click **"Capture point-cloud"** – an isolated `.ply` file is saved and displayed in an Open3D window.
* Repeat as needed; close the GUI with **Quit** or `Esc`.

---

## 🏗️ Using the CLI Scripts

### 1. Crop using a known AABB box

```bash
python isoler_ply.py -i scene_raw.ply -o piece.ply \
                     --bbox -0.05 0.12 -0.04 0.10 0.00 0.25
```

###2. Manual selection (polygon + click)

```bash
python isoler_ply.py -i scene_raw.ply --interactive
```

### 3. Fine-tune thresholds

```bash
python isoler_ply_tunable.py scene_raw.ply \
     --voxel 0.001 --eps 0.015 --min_pts 12 --plane_d 0.003
```

---

## ⚙️ Useful Parameters

| Parameter        | Default    | Description                                                |
|------------------|------------|------------------------------------------------------------|
| `--voxel`        | `0.003` m  | Voxel-grid size. `0` → no down-sampling.                   |
| `--eps`          | `0.01` m   | DBSCAN radius for density evaluation.                      |
| `--min_pts`      | `50`       | Min. number of neighbors in DBSCAN to form a cluster.      |
| `--plane_d`      | `0.004` m  | RANSAC tolerance for detecting the support plane.          |
| `--keep_normals` | off        | Preserve/recompute normals.                                |

---

## 🧐 Troubleshooting

| Symptom                        | Cause / Fix                                                                                                 |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| `No device connected`          | Missing USB 3.0 ➜ change port/cable, test with `realsense-viewer`.                                          |
| Black Open3D window            | Missing GPU/OpenGL driver ➜ update driver or run in `--headless` (write file only).                         |
| “No cluster detected”          | Object too small/sparse ➜ lower `--min_pts` or increase `--eps`.                                            |
| Object removed with plane      | Reduce `--plane_d` (2–3 mm) or disable RANSAC (remove the call).                                            |

---

## 📚 Resources / Docs

* Intel RealSense SDK 2.0 → [https://github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense)
* Open3D docs → [http://www.open3d.org/docs/](http://www.open3d.org/docs/)

---

## 📝 License

Project provided under the **MIT** license — use it well, contribute, share!

---

## AUTHORS
Darius Giannoli and Gabriel Taïeb

**Happy scanning!** 🕹️🎨
