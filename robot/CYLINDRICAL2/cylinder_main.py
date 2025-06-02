#!/usr/bin/env python3
"""
cylinder_main.py - Pipeline pour objets cylindriques
Détecte l'approche (top/body), extrait les dimensions et effectue le grasp
"""

import os
import sys
import re
import subprocess
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs

# ─── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# ─── Argument pour le .ply ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Cylinder grasp pipeline")
parser.add_argument(
    "ply",
    help="Chemin vers le fichier point-cloud (.ply) généré par la pipeline",
)
args = parser.parse_args()

# ─── Helpers ──────────────────────────────────────────────────────────────

def extract_cylinder_dimensions(ply_path):
    """Extrait les dimensions du cylindre en utilisant fitting.py"""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "fitting.py"), ply_path],
            capture_output=True, text=True, check=True, cwd=SCRIPT_DIR
        )
        
        # Parse les dimensions depuis la sortie
        lines = result.stdout.split('\n')
        radius = None
        height = None
        diameter = None
        
        for line in lines:
            if "Radius (r):" in line:
                radius = float(line.split(':')[1].strip())
            elif "Height (h):" in line:
                height = float(line.split(':')[1].strip())
            elif "Diameter:" in line:
                diameter = float(line.split(':')[1].strip())
        
        return radius, height, diameter
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Échec fitting.py: {e.stderr}", file=sys.stderr)
        return None, None, None

def detect_approach_method():
    """Détecte la méthode d'approche (top/body) avec ArUco"""
    return "body"

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"[CYLINDER] Traitement du fichier : {args.ply}")
    
    # 1) Extraire les dimensions du cylindre
    print("[CYLINDER] Extraction des dimensions du cylindre...")
    radius, height, diameter = extract_cylinder_dimensions(args.ply)
    
    if radius is None:
        print("[ERR] Impossible d'extraire les dimensions du cylindre", file=sys.stderr)
        sys.exit(1)
    
    print(f"[CYLINDER] Dimensions extraites : rayon={radius:.4f}m, hauteur={height:.4f}m, diamètre={diameter:.4f}m")
    
    # 2) Détecter la méthode d'approche
    approach = detect_approach_method()
    print(f"[CYLINDER] Méthode d'approche détectée : {approach}")
    
    # 3) Traitement selon l'approche
    if approach == "top":
        print("[CYLINDER] Approche 'top' - extraction du sommet...")
        try:
            subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "extract_top.py"), args.ply],
                cwd=SCRIPT_DIR, check=True
            )
            # Utiliser le diamètre pour le grasp top
            distance = diameter
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Échec extract_top.py: {e}", file=sys.stderr)
            sys.exit(1)
    else:  # body
        print("[CYLINDER] Approche 'body' - extraction du corps...")
        try:

            subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "extract_body.py"), args.ply],
                cwd=SCRIPT_DIR, check=True
)
            # Utiliser le diamètre pour le grasp body
            distance = diameter
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Échec extract_body.py: {e}", file=sys.stderr)
            sys.exit(1)
    
    # 4) Effectuer le grasp
    try:
        print(f"[CYLINDER] Exécution du grasp avec distance={distance:.4f}m, approche={approach}...")
        grasp_cmd = [
            sys.executable, 
            os.path.join(SCRIPT_DIR, "cylinder_grasp.py"), 
            "--distance", f"{distance:.4f}",
            "--grasp", approach
        ]
        
        result = subprocess.run(grasp_cmd, capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print(f"[CYLINDER] {result.stdout.strip()}")
        
        print("[CYLINDER] ✅ Séquence de grasp terminée avec succès")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Échec du grasp: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # 5) Attente de la touche 'r' pour remise à zéro
    print("Appuyez sur 'r' puis Entrée pour remettre le bras à zéro, ou sur une autre touche pour quitter.")
    choix = input("Votre choix : ").strip().lower()
    if choix == 'r':
        reset_path = os.path.join(SCRIPT_DIR, os.pardir, "reset.py")
        if not os.path.exists(reset_path):
            print(f"[ERR] Le fichier '{reset_path}' n'existe pas.", file=sys.stderr)
            sys.exit(1)
        print(f"[CYLINDER] Lancement de {reset_path} …")
        ret2 = subprocess.run([sys.executable, reset_path], check=False)
        if ret2.returncode != 0:
            print(f"[ERR] reset.py s'est terminé avec le code {ret2.returncode}", file=sys.stderr)
            sys.exit(ret2.returncode)
    else:
        print("[CYLINDER] Fin du programme.")

if __name__ == "__main__":
    main()