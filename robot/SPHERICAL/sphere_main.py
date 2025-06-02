from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sphere_dimensions import classify_point_cloud

def main() -> None:
    """Point d'entrée principal du pipeline de préhension sphérique."""

    # ── Fichier PLY par défaut ───────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    default_pc = script_dir / "sphere.ply"

    # ── Arguments CLI ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Détermine le type de prise pour un objet sphérique et appelle le contrôleur de robot."
    )
    parser.add_argument(
        "input_pc",
        nargs="?",
        default=str(default_pc),
        help=f"Chemin vers le nuage de points filtré (défaut : '{default_pc}').",
    )
    args = parser.parse_args()

    pc_path = Path(args.input_pc)
    if not pc_path.exists():
        parser.error(f"Le fichier point‑cloud '{pc_path}' n'existe pas.")

    # ── Analyse du nuage de points ───────────────────────────────────────
    center, radius, diameter, graspable, label = classify_point_cloud(str(pc_path))
    print(f"Diamètre calculé : {diameter:.4f} m")
    print(f"Préhensible : {graspable} (label : {label})")

    # Le classifieur renvoie déjà le type de prise : "spherical" ou "flat"
    grasp = label

    # ── Appel du script de prise finale ──────────────────────────────────
    cmd = [
        sys.executable,
        str(script_dir / "sphere_grasp.py"),
        "--grasp",
        grasp,
        "--distance",
        f"{diameter:.6f}",
    ]
    print("Exécution :", " ".join(cmd))

    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(
            f"Erreur : sphere_grasp.py s'est terminé avec le code {ret.returncode}",
            file=sys.stderr,
        )
        sys.exit(ret.returncode)

    # ── Interception de la touche 'r' pour remise à zéro ────────────────
    print("Appuyez sur 'r' puis Entrée pour remettre le bras à zéro, ou toute autre touche pour quitter.")
    choix = input("Votre choix : ").strip().lower()
    if choix == "r":
        reset_path = script_dir.parent / "reset.py"
        print(f"Lancement de {reset_path} …")
        ret2 = subprocess.run([sys.executable, str(reset_path)], check=False)
        if ret2.returncode != 0:
            print(
                f"Erreur : reset.py s'est terminé avec le code {ret2.returncode}",
                file=sys.stderr,
            )
            sys.exit(ret2.returncode)
    else:
        print("Fin du programme.")

if __name__ == "__main__":
    main()
