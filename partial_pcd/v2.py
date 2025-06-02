"""
realsense_gui_capture.py
------------------------
Aperçu live RealSense + bouton pour capturer le nuage de points isolé.
Les points sont colorés (rouge) et affichés sur fond noir.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import datetime, threading, sys
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

# ------------------------------------------------------------------
# 1. Initialisation RealSense
# ------------------------------------------------------------------
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile  = pipeline.start(config)

align_to     = rs.stream.color
align        = rs.align(align_to)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
o3d_intr = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

# ------------------------------------------------------------------
# 2. Tkinter GUI
# ------------------------------------------------------------------
root = tk.Tk()
root.title("RealSense – Aperçu live")
root.resizable(False, False)

label_img = ttk.Label(root)      # zone vidéo
label_img.pack(padx=5, pady=5)

btn_frame = ttk.Frame(root)      # boutons
btn_frame.pack(pady=(0, 5))

def quit_app():
    pipeline.stop()
    root.destroy()
    sys.exit()

ttk.Button(btn_frame, text="Quitter", command=quit_app).grid(row=0, column=1, padx=5)
btn_capture = ttk.Button(btn_frame, text="Capture point-cloud")
btn_capture.grid(row=0, column=0, padx=5)

# ------------------------------------------------------------------
# 3. Algo de nettoyage
# ------------------------------------------------------------------
def clean_pointcloud(pcd):
    """Supprime le plan principal, garde le plus gros cluster"""
    pcd = pcd.voxel_down_sample(0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    plane_model, inliers = pcd.segment_plane(0.004, 3, 800)
    pcd_no_plane = pcd.select_by_index(inliers, invert=True)

    labels = np.array(pcd_no_plane.cluster_dbscan(eps=0.01, min_points=50))
    if labels.size == 0 or (labels >= 0).sum() == 0:
        return None

    counts  = np.bincount(labels[labels >= 0])
    largest = counts.argmax()
    idx     = np.where(labels == largest)[0]
    return pcd_no_plane.select_by_index(idx)

# ------------------------------------------------------------------
# 4. Affichage Open3D avec fond noir
# ------------------------------------------------------------------
def show_cloud(cloud, title="Objet isolé"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])   # noir
    opt.point_size       = 5                       # un peu plus gros
    vis.run()
    vis.destroy_window()

# ------------------------------------------------------------------
# 5. Capture (dans un thread)
# ------------------------------------------------------------------
def do_capture(frames):
    aligned = align.process(frames)
    depth   = aligned.get_depth_frame()
    color   = aligned.get_color_frame()

    depth_img = o3d.geometry.Image(np.asanyarray(depth.get_data()))
    color_img = o3d.geometry.Image(np.asanyarray(color.get_data()))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img,
        depth_scale=1.0/depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
    pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

    obj = clean_pointcloud(pcd)
    if obj is None:
        print("⚠️  Aucun objet détecté, ajuste les paramètres.")
        return

    # ---- couleur plus visible ----
    obj.paint_uniform_color([1, 0, 0])    # rouge vif
    # ------------------------------

    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"objet_seul_{ts}.ply"
    o3d.io.write_point_cloud(name, obj)
    print(f"✅  Enregistré : {name} ({len(obj.points)} pts)")

    show_cloud(obj)

def on_capture():
    try:
        frames = pipeline.wait_for_frames()
    except RuntimeError as e:
        print("Erreur pipeline :", e)
        return
    threading.Thread(target=do_capture, args=(frames,), daemon=True).start()

btn_capture.config(command=on_capture)

# ------------------------------------------------------------------
# 6. Mise à jour vidéo en continu
# ------------------------------------------------------------------
def update_stream():
    try:
        frames  = pipeline.wait_for_frames(timeout_ms=1)
        color_f = frames.get_color_frame()
        if not color_f:
            root.after(1, update_stream); return
        frame_np = np.asanyarray(color_f.get_data())[:, :, ::-1]  # BGR→RGB
        imgtk    = ImageTk.PhotoImage(Image.fromarray(frame_np))
        label_img.imgtk = imgtk
        label_img.configure(image=imgtk)
    except Exception:
        pass
    root.after(1, update_stream)

root.after(0, update_stream)

# ------------------------------------------------------------------
# 7. Lancement GUI
# ------------------------------------------------------------------
if __name__ == "__main__":
    root.mainloop()
