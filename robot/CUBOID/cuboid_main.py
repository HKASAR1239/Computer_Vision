#!/usr/bin/env python3
"""
cuboid_main.py simplifié - Plus de sélection ArUco
Extrait directement les dimensions et effectue le grasp
Ajout de visualisations pour le nuage et les dimensions
"""

import os
import sys
import re
import subprocess
import argparse
import cv2

# ─── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# ─── Argument pour le .ply ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simplified cuboid grasp pipeline")
parser.add_argument(
    "ply",
    help="Chemin vers le fichier point-cloud (.ply) généré par la pipeline",
)
args = parser.parse_args()

# ─── Helpers ──────────────────────────────────────────────────────────────

def extract_dimensions_from_text(text):
    """
    Parse lines like "Length : 0.0580" → returns dict of dimensions and (width, length, height).
    """
    dims = {k.lower(): float(v) for k, v in re.findall(r"(\w+)\s*:\s*([\d\.]+)", text)}
    return dims, (dims.get("width", 0.0), dims.get("length", 0.0), dims.get("height", 0.0))


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"[CUBOID] Traitement du fichier : {args.ply}")
    
    # 1) Extraire les dimensions du nuage de points
    try:
        print("[CUBOID] Extraction des dimensions...")
        dims_out = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "cuboid_dimensions.py"), args.ply],
            capture_output=True, text=True, check=True
        ).stdout
        
        dims_named, (w, l, h) = extract_dimensions_from_text(dims_out)
        print(f"[CUBOID] Dimensions extraites : width={w:.3f}m, length={l:.3f}m, height={h:.3f}m")
        

        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Échec de l'extraction des dimensions : {e.stderr}", file=sys.stderr)
        sys.exit(1)
    
    # 2) Effectuer le grasp avec la longueur
    try:
        print(f"[CUBOID] Exécution du grasp avec length={l:.3f}m...")
        grasp_cmd = [
            sys.executable, 
            os.path.join(SCRIPT_DIR, "cuboid_grasp.py"), 
            "--length", f"{l:.3f}"
        ]
        
        result = subprocess.run(grasp_cmd, capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print(f"[CUBOID] {result.stdout.strip()}")
        
        print("[CUBOID] ✅ Séquence de grasp terminée avec succès")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Échec du grasp : {e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

    # ── Attente de la touche 'r' pour remise à zéro ─────────────────────────
    print("Appuyez sur 'r' pour remettre le bras à zéro, ou toute autre touche pour quitter.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('r'):
        reset_path = os.path.join(SCRIPT_DIR, os.pardir, "reset.py")
        if not os.path.exists(reset_path):
            print(f"[ERR] Le fichier '{reset_path}' n'existe pas.", file=sys.stderr)
            sys.exit(1)
        print(f"[CUBOID] Lancement de {reset_path} …")
        ret2 = subprocess.run([sys.executable, reset_path], check=False)
        if ret2.returncode != 0:
            print(f"[ERR] reset.py s'est terminé avec le code {ret2.returncode}", file=sys.stderr)
            sys.exit(ret2.returncode)
    else:
        print("[CUBOID] Fin du programme.")

if __name__ == "__main__":
    main()