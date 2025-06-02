from __future__ import annotations

import sys
import os
import datetime
import importlib.util
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import time

import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌  Ultralytics n'est pas installé : pip install ultralytics")

# ────────────────────── Configuration ────────────────────────────────
MIN_POINTS_THRESHOLD = 100   
SAVE_DEBUG_CLOUDS = True     
# ────────────────────── Chemins projet ────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
for sub in ("partial_pcd", "shape_identification"):
    p = BASE_DIR / sub
    if p.exists():
        sys.path.insert(0, str(p))

# ────────────────────── Loader utilitaire ─────────────────────────────

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ────────────────────── YOLO Object Selector ──────────────────────────

class YOLOSelector:
    """Sélecteur d'objets basé sur YOLO"""
    
    def __init__(self):
        print("[INFO] Chargement modèle YOLO pour sélection d'objets...")
        self.model = YOLO('yolov8n.pt')  
        
        # Configuration
        self.ignored_classes = ["dining table", "person"]  # Classes à ignorer
        self.confidence_threshold = 0.7
        self.camera_angle = "face"  
        
        # État de sélection
        self.selected_box = None
        self.selected_class = None
        self.selected_confidence = 0.0
        
        print(f"[YOLO] Modèle chargé. Mode: {self.camera_angle}")
    
    def set_camera_angle(self, angle: str):
        """Change l'angle de la caméra (face/gauche/droite)"""
        if angle in ["face", "gauche", "droite"]:
            self.camera_angle = angle
            print(f"[YOLO] Mode changé: {angle}")
        else:
            print(f"[YOLO] Angle invalide: {angle}")
    
    def cycle_camera_angle(self):
        """Cycle entre les angles face -> gauche -> droite -> face"""
        angles = ["face", "gauche", "droite"]
        current_idx = angles.index(self.camera_angle)
        next_idx = (current_idx + 1) % len(angles)
        self.set_camera_angle(angles[next_idx])
    
    def process(self, frame: np.ndarray):
        """Traite une frame pour détecter et sélectionner un objet"""
        height, width = frame.shape[:2]
        center_x = width // 2
        
        # Détection YOLO
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        
        if boxes is None:
            self.selected_box = None
            return
        
        # Filtrer les détections
        filtered_boxes = []
        for box in boxes:
            confidence = box.conf.item()
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            # Ignorer les classes non pertinentes et les détections avec faible confiance
            if class_name not in self.ignored_classes and confidence >= self.confidence_threshold:
                filtered_boxes.append(box)
        
        # Sélection selon l'angle de caméra
        self.selected_box = None
        
        if len(filtered_boxes) > 0:
            if self.camera_angle == "face":
                # Sélectionner l'objet le plus centré horizontalement
                min_distance_to_center = float('inf')
                for box in filtered_boxes:
                    x1, _, x2, _ = map(int, box.xyxy[0])
                    box_center_x = (x1 + x2) // 2
                    distance_to_center = abs(box_center_x - center_x)
                    if distance_to_center < min_distance_to_center:
                        min_distance_to_center = distance_to_center
                        self.selected_box = box
            
            elif self.camera_angle == "gauche":
                # Sélectionner l'objet le plus à gauche
                min_x = float('inf')
                for box in filtered_boxes:
                    x1, _, _, _ = map(int, box.xyxy[0])
                    if x1 < min_x:
                        min_x = x1
                        self.selected_box = box
            
            elif self.camera_angle == "droite":
                # Sélectionner l'objet le plus à droite
                max_x = -float('inf')
                for box in filtered_boxes:
                    _, _, x2, _ = map(int, box.xyxy[0])
                    if x2 > max_x:
                        max_x = x2
                        self.selected_box = box
        
        # Mettre à jour les informations de l'objet sélectionné
        if self.selected_box is not None:
            self.selected_confidence = self.selected_box.conf.item()
            class_id = int(self.selected_box.cls)
            self.selected_class = self.model.names[class_id]
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Dessine l'overlay avec l'objet sélectionné"""
        overlay = frame.copy()
        
        # Dessiner l'objet sélectionné
        if self.selected_box is not None:
            x1, y1, x2, y2 = map(int, self.selected_box.xyxy[0])
            
            # Rectangle de sélection (vert)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Label avec classe et confiance
            label = f"{self.selected_class}: {self.selected_confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Afficher le mode actuel et les instructions
        mode_text = f"Mode: {self.camera_angle.upper()} | 'a': changer | 'c': capturer"
        cv2.putText(overlay, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Afficher le statut de sélection
        if self.selected_box is not None:
            status_text = f"OBJET SELECTIONNE: {self.selected_class}"
            color = (0, 255, 0)  # Vert
        else:
            status_text = "Aucun objet sélectionné (capture manuelle possible)"
            color = (0, 255, 255)  # Jaune
        
        cv2.putText(overlay, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay
    
    def has_selection(self) -> bool:
        """Vérifie si un objet est sélectionné"""
        return self.selected_box is not None
    
    def get_selection_info(self) -> Tuple[str, float]:
        """Retourne les informations de l'objet sélectionné"""
        if self.selected_box is not None:
            return self.selected_class, self.selected_confidence
        return "Aucun", 0.0

# ────────────────────── shape_id ────────────────
shape_mod = load_module("shape_id", BASE_DIR / "shape_identification" / "shape_id.py")
classify_shape = shape_mod.classify  # type: ignore[attr-defined]

import numpy as _np
if not hasattr(shape_mod, "_np2_patch"):
    def _tube_stats_np2(pts_c: _np.ndarray, vec: _np.ndarray):
        t = pts_c @ vec
        r = _np.linalg.norm(pts_c - _np.outer(t, vec), axis=1)
        m = r.mean() or 1e-6
        return r.std() / m, _np.ptp(t) / (2 * m)
    shape_mod.tube_stats = _tube_stats_np2  # type: ignore[attr-defined]
    shape_mod._np2_patch = True

# ────────────────────── Modèle poignée YOLOv9 ─────────────────────────
HANDLE_MODEL = (BASE_DIR / "handle_detection" / "cup_handle" / "runs" / "detect" /
                "train4" / "weights" / "best.pt")
if not HANDLE_MODEL.exists():
    sys.exit(f"❌  Modèle poignée introuvable : {HANDLE_MODEL}")
print("[INFO] Chargement modèle poignée…")
yolo_handle = YOLO(str(HANDLE_MODEL))
YOLO_DEVICE = "mps" if sys.platform == "darwin" else "cpu"
YOLO_CONF   = 0.4

# ────────────────────── RealSense ─────────────────────────────────────
print("[INFO] Initialisation RealSense …")
pipeline, config = rs.pipeline(), rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align        = rs.align(rs.stream.color)
depth_scale  = profile.get_device().first_depth_sensor().get_depth_scale()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
o3d_intr = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

# ────────────────────── Initialisation sélecteur YOLO ────────────────
selector = YOLOSelector()

# ────────────────────── Helpers ───────────────────────────────────────

def detect_handle(img: np.ndarray, obj_class: str) -> Tuple[bool, np.ndarray | None]:
    """Renvoie (has_handle, annotated_img). Si aucune poignée ou non-'cup' : annotated_img=None."""
    res = yolo_handle.predict(img, device=YOLO_DEVICE, imgsz=640,
                              conf=YOLO_CONF, verbose=False)[0]
    if res.boxes is None or res.boxes.cls is None:
        return False, None
    handle_ids = [i for i, name in res.names.items() if "handle" in name.lower()]
    has = any(int(c) in handle_ids for c in res.boxes.cls.cpu().numpy())
    # Poignée valide uniquement si objet est une "cup"
    if has and obj_class.lower() == "cup":
        return True, res.plot()
    return False, None


def clean_cloud(pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
    pcd = pcd.voxel_down_sample(0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    _, plane = pcd.segment_plane(0.004, 3, 800)
    pcd = pcd.select_by_index(plane, invert=True)
    labels = _np.array(pcd.cluster_dbscan(eps=0.01, min_points=30))
    if labels.size == 0:
        return None
    return pcd.select_by_index(_np.where(labels == _np.bincount(labels[labels >= 0]).argmax())[0])


def get_pointcloud(frames: rs.composite_frame) -> Optional[o3d.geometry.PointCloud]:
    aligned = align.process(frames)
    d, c = aligned.get_depth_frame(), aligned.get_color_frame()
    if not d or not c:
        return None
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(_np.asanyarray(c.get_data())),
        o3d.geometry.Image(_np.asanyarray(d.get_data())),
        depth_scale=1.0 / depth_scale, depth_trunc=3.0, convert_rgb_to_intensity=False)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
    cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return clean_cloud(cloud)


def view_cloud(cloud: o3d.geometry.PointCloud, title: str):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 920, 720)
    cloud.paint_uniform_color([1, 0, 0])
    opt = vis.get_render_option(); opt.background_color, opt.point_size = _np.zeros(3), 5
    vis.add_geometry(cloud); vis.run(); vis.destroy_window()


# ────────────────────── Boucle principale ────────────────────────────
cv2.namedWindow("RealSense - YOLO Selection", cv2.WINDOW_NORMAL)
print("[INFO] c : capture (avec ou sans sélection YOLO) | a : changer angle | q : quit")
subprocess.run(["python", "torque_ini.py"])
print("torque activé")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame_bgr = _np.asanyarray(color_frame.get_data())

        # Mise à jour sélection YOLO + overlay
        selector.process(frame_bgr)
        overlay = selector.draw_overlay(frame_bgr.copy())
        cv2.imshow("RealSense - YOLO Selection", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            selector.cycle_camera_angle()
        elif key == ord('c'):
            # ─── INFORMATION SÉLECTION (optionnelle) ────────────────────────
            if selector.has_selection():
                obj_class, obj_conf = selector.get_selection_info()
                print(f"[CAPTURE] Objet YOLO sélectionné: {obj_class} (confiance: {obj_conf:.2f})")
            else:
                obj_class, obj_conf = "Manuel", 0.0
                print("[CAPTURE] Aucun objet YOLO sélectionné - capture manuelle")
            
            # ─── CAPTURE UNIQUE ───────────────────────────────────────────
            # Capture pour la détection de poignée
            cap_frames = pipeline.wait_for_frames()
            color_cap = cap_frames.get_color_frame()
            if not color_cap:
                print("[WARN] Frame couleur manquante.")
                continue
            frame_cap = _np.asanyarray(color_cap.get_data())

            print("[CAPTURE] Détection poignée...")
            has_handle, annotated = detect_handle(frame_cap, obj_class)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Sauvegarde JPEG si poignée
            if annotated is not None:
                jpg = BASE_DIR / f"handle_{ts}.jpg"
                cv2.imwrite(str(jpg), annotated)
                print(f"🖼  Image poignée enregistrée : {jpg.name}")

            # ─── CLASSIFICATION SUR UN SEUL NUAGE ─────────────────────────
            print("[CAPTURE] Capture nuage de points...")
            cloud = get_pointcloud(cap_frames)
            
            if cloud is None or len(cloud.points) < MIN_POINTS_THRESHOLD:
                print("[WARN] Échec capture nuage ou trop peu de points.")
                continue

            shape = classify_shape(_np.asarray(cloud.points))
            print(f"[CAPTURE] Forme détectée: {shape} ({len(cloud.points)} points)")

            # Sauvegarde du nuage
            ply = BASE_DIR / f"objet_{ts}.ply"
            o3d.io.write_point_cloud(str(ply), cloud)
            print(f"✅  Nuage enregistré : {ply.name} ({len(cloud.points)} points)")

            print(f"🔎  Objet: {obj_class} | Poignée : {'Oui' if has_handle else 'Non'} | Forme : {shape}")

            # Affichage du nuage
            title = f"{obj_class} - {shape} – Poignée : {'Oui' if has_handle else 'Non'}"
            if obj_class == "Manuel":
                title = f"{shape} – Poignée : {'Oui' if has_handle else 'Non'} (Capture manuelle)"

            ####ICI POUR AJOUTER LA LOGIQUE DE CHANGER######
            view_cloud(cloud, title)

            # ─── CONFIRMATION AVANT DE CONTINUER ───────────────────
            resp = input("Fermez la fenêtre. Continuer la pipeline ? [y/N] : ").strip().lower()
            print(f"Debug: received '{resp}', length={len(resp)}, repr={repr(resp)}")
            if 'n' in resp :
                print("Pipeline interrompue par l'utilisateur.\n")
                continue  # repart au début de la boucle principale

            # ─── OVERRIDE MANUEL DE LA FORME ───────────────────────
            override = input(f"Forme détectée = '{shape}'. Presser Entrée pour valider ou taper une autre forme : ").strip()
            if override:
                print(f"→ Forme remplacée : '{override}'\n")
                shape = override

            #########

            # ─── ROUTAGE ROBOT (utilise shape) ─────────────────────────
            shape_l = shape.lower()
            robot_dir = BASE_DIR / "robot"
            
            if has_handle:                         
                robot_main = robot_dir / "hook_main.py"
                label = "hook"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} introuvable ⇒ branche {label} ignorée.")
                else:
                    print(f"[PIPE] Forme {label} détectée → lancement {robot_main.name} …")
                    try:
                        # Passer l'image de la poignée en argument
                        cmd = [sys.executable, str(robot_main), str(ply)]
                        if annotated is not None:
                            cmd.extend(["--handle_image", str(BASE_DIR / f"handle_{ts}.jpg")])
                        subprocess.run(cmd, cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} a échoué : {e}", file=sys.stderr)
                        

            elif shape_l == "cuboid":
                robot_main = robot_dir / "CUBOID2" / "cuboid_main.py"
                label = "cuboid"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} introuvable ⇒ branche {label} ignorée.")
                else:
                    print(f"[PIPE] Forme {label} détectée → lancement {robot_main.name} …")
                    try:
                        subprocess.run([sys.executable, str(robot_main), str(ply)], cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} a échoué : {e}", file=sys.stderr)
                        
            elif shape_l == "cylindrique":
                robot_main = robot_dir / "CYLINDRICAL" / "cylinder_main.py"
                label = "cylinder"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} introuvable ⇒ branche {label} ignorée.")
                else:
                    print(f"[PIPE] Forme {label} détectée → lancement {robot_main.name} …")
                    try:
                        subprocess.run([sys.executable, str(robot_main), str(ply)], cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} a échoué : {e}", file=sys.stderr)
                        
            elif shape_l == "spherique":
                robot_main = robot_dir / "SPHERICAL2" / "sphere_main.py"
                label = "sphere"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} introuvable ⇒ branche {label} ignorée.")
                else:
                    print(f"[PIPE] Forme {label} détectée → lancement {robot_main.name} …")
                    try:
                        subprocess.run([sys.executable, str(robot_main), str(ply)], cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} a échoué : {e}", file=sys.stderr)

            else:
                print(f"[INFO] Forme {shape_l} non reconnue, aucune action robot.")

except KeyboardInterrupt:
    pass

finally:
    print("[INFO] Arrêt RealSense…")
    pipeline.stop()
    cv2.destroyAllWindows()