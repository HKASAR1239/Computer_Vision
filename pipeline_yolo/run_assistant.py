import time, queue, sys, json
from pathlib import Path
import numpy as np, torch, sounddevice as sd
from utils import to_logmelspec, pad_or_trim, SAMPLE_RATE
from models import WakeNet, CmdNet

# ---------- paramètres ----------
WAKE_THRESHOLD   = 0.99        # tune si faux positifs
CHUNK_SEC        = 0.5          # analyse toutes les 0.5 s
POST_WAKE_SEC    = 2.2          # temps maximal pour la commande
DEVICE           = None         # default input
# --------------------------------

wakenet = WakeNet();  wakenet.load_state_dict(torch.load('models/wakenet.pt')); wakenet.eval()
cmdnet  = CmdNet(n_classes := len(json.loads(Path('models/idx2cmd.json').read_text())) )
cmdnet.load_state_dict(torch.load('models/cmdnet.pt')); cmdnet.eval()
idx2cmd = json.loads(Path('models/idx2cmd.json').read_text())

q_audio = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status: print(status, file=sys.stderr)
    q_audio.put(indata.copy())

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=int(CHUNK_SEC*SAMPLE_RATE),
    channels=1,
    dtype='float32',
    callback=audio_callback,
    device=DEVICE,
)

def detect_wake(buf: np.ndarray) -> bool:
    wav = torch.from_numpy(buf.T).float()
    feat = to_logmelspec(wav)
    p = wakenet(feat.unsqueeze(0)).item()
    return p > WAKE_THRESHOLD

def recognise_cmd():
    buf = []
    timeout = time.time()+POST_WAKE_SEC
    while time.time()<timeout:
        try:
            chunk = q_audio.get(timeout=POST_WAKE_SEC)
            buf.append(chunk)
        except queue.Empty:
            break
    if not buf: return None
    wav = torch.from_numpy(np.concatenate(buf).T).float()
    feat = to_logmelspec(pad_or_trim(wav, int(2*SAMPLE_RATE)))
    logits = cmdnet(feat.unsqueeze(0))
    cmd = idx2cmd[str(logits.argmax(-1).item())]
    return cmd

print("👂  Assistant actif – dites votre wake-word…")
with stream:
    ring = np.zeros(int(1.5*SAMPLE_RATE), dtype=np.float32)
    while True:
        chunk = q_audio.get()
        ring = np.roll(ring, -len(chunk)); ring[-len(chunk):] = chunk[:,0]
        if detect_wake(ring):
            print("⚡ Wake-word détecté ! Dites une commande…")
            cmd = recognise_cmd()
            if cmd:
                print(f"🗣️  →  « {cmd} »")
            else:
                print("🤔  Aucune commande détectée.")
