from pathlib import Path
import argparse, sounddevice as sd, soundfile as sf, numpy as np, sys, time

# ----------- constantes ----------- #
SAMPLE_RATE   = 16_000
CMD_DURATION  = 2.0        
NUM_SAMPLES   = 50         
# ---------------------------------- #

def record(seconds: float) -> np.ndarray:
    print(f"[REC] {seconds:.1f}s…")
    buf = sd.rec(int(seconds * SAMPLE_RATE),
                 samplerate=SAMPLE_RATE,
                 channels=1,
                 dtype='int16')
    sd.wait()
    return buf.squeeze()

def save_wav(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, arr, SAMPLE_RATE, subtype="PCM_16")

    # Essaye d’afficher un chemin relatif ; sinon tombe gracieusement.
    try:
        rel = path.resolve().relative_to(Path.cwd())
        print(f"  ↳ {rel}")
    except ValueError:
        print(f"  ↳ {path}")

def main():
    ap = argparse.ArgumentParser(description="Recorder commandes uniquement.")
    ap.add_argument("--file", required=True,
                    help="Fichier texte : une commande par ligne.")
    ap.add_argument("--out", default="dataset",
                    help="Répertoire racine (défaut: ./dataset)")
    ap.add_argument("--num", type=int, default=NUM_SAMPLES,
                    help=f"Prises par commande (défaut: {NUM_SAMPLES})")
    ap.add_argument("--duration", type=float, default=CMD_DURATION,
                    help=f"Durée (s) de chaque prise (défaut: {CMD_DURATION})")
    args = ap.parse_args()

    cmds = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not cmds:
        sys.exit("❌  Aucune commande trouvée dans le fichier.")

    root = Path(args.out)
    print(f"\nEnregistrement vers : {root.resolve()}")
    print(f"Commandes : {cmds}\n")

    try:
        for cmd in cmds:
            cmd_dir = root / "commands" / cmd.replace(" ", "_")
            for i in range(args.num):
                input(f"Commande « {cmd} » ({i+1}/{args.num}) – appuyez sur Entrée puis parlez…")
                audio = record(args.duration)
                save_wav(audio, cmd_dir / f"sample_{i:03d}.wav")
                time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu par l'utilisateur.")

    print("\n✅ Terminé ! Vous pouvez lancer preprocess.py puis train_commands.py.")

if __name__ == "__main__":
    main()
