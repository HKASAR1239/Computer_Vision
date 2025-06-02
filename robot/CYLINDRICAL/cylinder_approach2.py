#!/usr/bin/env python3
"""cylinder_approach.py — v3 (2025‑05‑08)

Décide entre la saisie **top** (axiale) et **body** (latérale) d’un
cylindre à l’aide de deux marqueurs ArUco (DICT_4X4_50).

Correctifs v3 :
  • **Attente démarrage caméra** : on ignore les frames ratées au lieu de
    quitter, ce qui laissait l’algorithme conclure immédiatement « body ».
  • **Sortie "unknown"** si aucun vote n’a été collecté (marqueurs non
    détectés).
  • Paramètre `--min-votes` (défaut 5) : assure un minimum d’observations
    avant de décider, quitte à prolonger légèrement la durée.

Usage rapide :
    python cylinder_approach.py --debug

Options principales :
  --duration 2.0   temps *max* (s) de collecte après la 1ʳᵉ frame valide
  --min-votes 5    nb minimal de votes pour conclure
  --ratio 1.2      critère vertical dominant (|dy| > ratio·|dx| ⇒ top)
  --min-disp 3     filtre les micro‑vibrations (px)
  --debug          fenêtre OpenCV avec les votes

Si aucun vote n’est obtenu avant `--duration`, le script affiche
**"unknown"** et renvoie le code de sortie 1.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="cylinder_approach")
    ap.add_argument("--body-id", type=int, default=0,
                    help="ID ArUco collé sur le cylindre (immobile)")
    ap.add_argument("--tool-id", type=int, default=1,
                    help="ID ArUco sur l'outil qui s'approche")
    ap.add_argument("--camera", type=int, default=0,
                    help="Index caméra OpenCV")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Durée max de collecte APRÈS 1ʳᵉ frame valide (s)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="FPS max pour économiser le CPU")
    ap.add_argument("--min-votes", type=int, default=5,
                    help="Nombre minimal de votes avant décision")
    ap.add_argument("--ratio", type=float, default=1.2,
                    help="|dy| > ratio*|dx| ⇒ vote top")
    ap.add_argument("--min-disp", type=float, default=3.0,
                    help="Δx ou Δy min (px) pour enregistrer un vote")
    ap.add_argument("--overhead", action="store_true",
                    help="Caméra en plongée (y inversé)")
    ap.add_argument("--debug", action="store_true",
                    help="Fenêtre de debug OpenCV")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def detect_centers(frame: np.ndarray,
                   detector: cv2.aruco.ArucoDetector,
                   body_id: int,
                   tool_id: int) -> Tuple[np.ndarray, np.ndarray] | None:
    """Return pixel centres (cx, cy) of body and tool markers in *frame*.
    Returns None if one of the markers is missing or no markers."""
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    ids_flat = ids.flatten()
    if body_id not in ids_flat or tool_id not in ids_flat:
        return None

    idx_body = np.where(ids_flat == body_id)[0][0]
    idx_tool = np.where(ids_flat == tool_id)[0][0]
    center_body = corners[idx_body][0].mean(axis=0)
    center_tool = corners[idx_tool][0].mean(axis=0)
    return center_body, center_tool


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    # --- ArUco detector -----------------------------------------------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # --- Camera --------------------------------------------------------------------------
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit("[ERROR] Impossible d'ouvrir la caméra ")

    votes = Counter()
    first_valid_time: float | None = None  # will be set after 1st valid frame
    frame_period = 1.0 / args.fps

    last_tool: np.ndarray | None = None

    try:
        while True:
            tic = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)  # laisse la caméra démarrer
                continue  # ne pas quitter → attendre qu'elle s'allume

            centers = detect_centers(frame, detector, args.body_id, args.tool_id)
            if centers is None:
                # Caméra OK mais marqueurs pas encore visibles.
                if args.debug:
                    cv2.imshow("cylinder_approach", frame)
                    cv2.waitKey(1)
                continue

            # From here, we have both markers
            if first_valid_time is None:
                first_valid_time = time.time()

            center_body, center_tool = centers

            if last_tool is not None:
                dx = center_tool[0] - last_tool[0]
                dy = center_tool[1] - last_tool[1]

                # Ignore tiny jitter
                if max(abs(dx), abs(dy)) >= args.min_disp:
                    if args.overhead:
                        dy = -dy
                    if abs(dy) > args.ratio * abs(dx):
                        votes["top"] += 1
                        current = "TOP"
                    else:
                        votes["body"] += 1
                        current = "BODY"
            else:
                dx = dy = 0  # for debug display only
            last_tool = center_tool

            # ------------------------------ DEBUG ---------------------------------------
            if args.debug:
                cv2.aruco.drawDetectedMarkers(frame, [], None)  # draw all detected by default
                cv2.putText(frame, f"dx={dx:+.1f} dy={dy:+.1f}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"votes → top:{votes['top']} body:{votes['body']}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("cylinder_approach", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break  # Esc
            # ---------------------------------------------------------------------------

            # Stopping conditions --------------------------------------------------------
            if first_valid_time is not None:
                if (time.time() - first_valid_time) >= args.duration and (votes['top'] + votes['body']) >= args.min_votes:
                    break

            # FPS control
            delay = frame_period - (time.time() - tic)
            if delay > 0:
                time.sleep(delay)

    finally:
        cap.release()
        if args.debug:
            cv2.destroyAllWindows()

    total_votes = votes['top'] + votes['body']
    if total_votes == 0:
        print("unknown")
        sys.exit(1)

    grasp = "top" if votes['top'] >= votes['body'] else "body"
    print(grasp)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
