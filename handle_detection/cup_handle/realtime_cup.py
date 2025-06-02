#!/usr/bin/env python3
"""
Real-time cup-handle detection on macOS (M-series) with YOLOv9
--------------------------------------------------------------
Requirements:
    • Python 3.8+
    • ultralytics (pip install ultralytics)
    • opencv-python (pip install opencv-python)

Usage examples:
    # avec les valeurs par défaut
    python detect_cup_handle.py

    # spécifier un autre modèle et cam_id
    python detect_cup_handle.py \
        --model runs/detect/exp/weights/best.pt \
        --cam-id 1 \
        --device cpu \
        --conf-thres 0.4 \
        --img-size 512

Arguments:
    --model       Chemin vers le fichier .pt (défaut: runs/detect/train4/weights/best.pt)
    --cam-id      Identifiant de la caméra (défaut: 0)
    --device      "cpu", "mps" ou "cuda" (défaut: mps)
    --conf-thres  Seuil de confiance [0.0–1.0] (défaut: 0.5)
    --img-size    Taille d’entrée du modèle (défaut: 640)
    --show-stats  Afficher le nombre total de frames traitées à la fin
    -h, --help    Aide

Appuie sur « q » pour quitter.
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Détection cup-handle en temps réel")
    p.add_argument("--model",       type=Path,
                                 default=Path("runs/detect/train4/weights/best.pt"),
                                 help="chemin vers le modèle .pt")
    p.add_argument("--cam-id",      type=int, default=0,
                                 help="ID de la caméra (0, 1, ...)")
    p.add_argument("--device",      type=str, default="mps",
                                 choices=["cpu","mps","cuda"],
                                 help="device pour l’inférence")
    p.add_argument("--conf-thres",  type=float, default=0.5,
                                 help="seuil de confiance [0.0–1.0]")
    p.add_argument("--img-size",    type=int, default=640,
                                 help="résolution de l’image (carrée)")
    p.add_argument("--show-stats",  action="store_true",
                                 help="afficher le nombre de frames total à la fin")
    return p.parse_args()


def main():
    args = parse_args()

    # Vérification du modèle
    if not args.model.exists():
        sys.exit(f"❌  Modèle introuvable : {args.model}")

    print(f"[INFO] Chargement du modèle depuis : {args.model}")
    model = YOLO(str(args.model))

    print(f"[INFO] Ouverture de la caméra ID {args.cam_id}…")
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        sys.exit(f"❌  Impossible d’ouvrir la caméra ID {args.cam_id}")

    window_name = "Cup Handle Detector (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    t_start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame non lue, arrêt.")
                break

            # Mesure temps d’inférence
            t0 = time.time()
            results = model.predict(
                source=frame,
                device=args.device,
                imgsz=args.img_size,
                conf=args.conf_thres,
                verbose=False
            )
            inf_time = (time.time() - t0) * 1000  # ms

            # Dessin des boîtes
            annotated = results[0].plot()

            # Overlay texte : FPS & temps inférence
            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            text = f"FPS: {fps:.1f} | Inf: {inf_time:.1f} ms | Conf>={args.conf_thres}"
            cv2.putText(annotated, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Récupération des classes détectées
            classes = results[0].boxes.cls
            if len(classes):
                cls_names = [model.names[int(c)] for c in classes]
                unique = set(cls_names)
                stats = " | ".join(f"{u}:{cls_names.count(u)}" for u in unique)
                cv2.putText(annotated, stats, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            cv2.imshow(window_name, annotated)

            # Quitte avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - t_start
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"[INFO] Total frames : {frame_count}")
        print(f"[INFO] Temps écoulé : {total_time:.1f}s")
        print(f"[INFO] FPS moyen : {avg_fps:.1f}")

        if args.show_stats:
            print("=== DÉTECTION TERMINÉE ===")
            print(f"Frames traitées : {frame_count}")
            print(f"Durée totale     : {total_time:.1f} s")
            print(f"FPS moyen        : {avg_fps:.1f}")

if __name__ == "__main__":
    main()
