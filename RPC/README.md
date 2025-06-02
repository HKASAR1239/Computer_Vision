# Rock‑Paper‑Scissors CV (Npulse)

Real‑time hand‑gesture recognition with an AI “prediction”.  
Wave your hand under a camera, get the gesture (✊ ROCK / ✋ PAPER / ✌️ SCISSORS) and a suggested counter‑move calculated in real time.

* **rps_cam.py** – manual mode  
  *Press **Space** or click **Start round** to trigger a 3‑2‑1 countdown.*

* **rpc_auto.py** – fully automatic loop  
  *The countdown starts automatically, shows a recommendation for 2 s and restarts.*

* **rpc_mit.py** – same UI as `rps_cam.py` but with an enhanced “Iocaine‑lite” coach inspired by the MIT winning algorithm (1999 RoShamBo competition)


---

## 📽 Demo GIF
![Demo animation](../docs/demo_rpc.gif)



---
## 1 · Features

| Component          | Description                                                                  |
|--------------------|------------------------------------------------------------------------------|
| **MediaPipe Hands**| Fast 21‑landmark hand tracking on CPU (≈ 60 FPS on Apple M‑series).          |
| **Gesture logic**  | Heuristic finger‑count → ROCK / PAPER / SCISSORS.                            |
| **Markov coach**   | Learns \(P(\text{next}\mid\text{current})\) online and suggests the beat‑move.|
| **Iocaine‑lite**   | 18 meta‑strategies (Freq, Last, Markov 1 × second‑guessing) with exponential scoring.|
| **PySimpleGUI**    | Cross‑platform GUI; single window with live video, countdown and hints.      |
| **Countdown**      | 3‑2‑1 overlay, result freeze, automatic restart (in `rpc_auto.py`).          |
| **Raspberry Pi 5** | Works on aarch64; ~25 FPS at 640×480 with Pi camera or USB webcam.           |
| **Intel RealSense**| Optional support – pass `--realsense` flag (only in `rps_cam.py`).           |

<br>

## 2 · Folder layout

```
rpc/
├── requirements.txt      # Python wheels (CPU only)
├── setup_rps.sh          # one‑shot installer for macOS (Homebrew + pyenv + venv)
├── rps_cam.py            # manual countdown + Markov coach
├── rpc_auto.py           # auto‑loop countdown + Markov coach
├── rpc_mit.py            # manual countdown + Iocaine‑lite coach
└── setup/                # (optional) RealSense / Pi specific notes
```

> All scripts run on **macOS Sequoia / Apple Silicon** and **Raspberry Pi 5** without modification.

<br>

## 3 · Quick start (macOS, Apple Silicon)

```bash
git clone https://github.com/DariusGiannoli/CV-Npulse.git && cd rpc

# one‑shot installer – creates ~/.pyenv Python, venv & installs wheels
chmod +x setup_rps.sh
./setup_rps.sh

# activate the virtual‑env
source venv/bin/activate

# manual mode (press Space each round)
python rps_cam.py

#starting pose to start the countdown (rock over paper)
python rpc_auto.py

# MIT‑style coach (manual trigger)
python rpc_mit.py
```

<br>

## 4 · Quick start (Raspberry Pi 5, 64‑bit)

```bash
sudo apt update && sudo apt install python3-venv python3-opencv
python3 -m venv venv && source venv/bin/activate

pip install -r requirements.txt            # mediapipe wheels exist for aarch64

python rps_cam.py --width 640 --height 480 # Pi Camera or USB webcam
```

<br>

## 5 · Command‑line options

| Option                    | Default | Description                                |
|---------------------------|---------|--------------------------------------------|
| `--camera <index>`        | `0`     | OpenCV device index (0, 1, …)              |
| `--backend avf` (macOS)   | `any`   | Force AVFoundation backend                 |
| `--width <px>` `--height` | 1280×720| Capture resolution                         |
| `--realsense`             | off     | Use Intel RealSense D435i (only rps_cam.py)|

*On Raspberry Pi you can stick to 640×480 for best FPS.*

<br>

## 6 · How the algorithms work

### Markov 1 (default)

1. Keep a 3×3 matrix *C* where `C[i,j]` counts transitions *i → j*  
   (*i,j* ∈ {ROCK, PAPER, SCISSORS}).  
2. At each round, predict the most likely *next* gesture from the last one.  
3. Suggest the move that beats that prediction.

### Iocaine‑lite (MIT)

* 3 basic predictors  
  * global frequency, last‑move repeat, Markov 1*  
* For each, 6 meta‑strategies (beat / copy / lose × instant / 1‑step lag).  
* Exponential scoring (α ≈ 0.97) picks the best meta every round.  
* Typical win rate vs humans **55–60 %** (vs 45–50 % for simple Markov).

<br>

## 7 · Troubleshooting

| Problem / Symptom                                      | Fix |
|--------------------------------------------------------|-----|
| “`Camera grab failed`”                                 | Make sure no other app is using the webcam, or try `--camera 1` / `--backend avf` (macOS). |
| PySimpleGUI raises `tkinter`‑related errors            | Run `brew install tcl-tk`, then use `setup_rps.sh` which compiles Python with Tk. |
| Landmarks appear but gesture is **UNKNOWN**            | Ensure all 5 fingers are plainly visible; use brighter light; reduce resolution to 640×480. |
| Always suggests **PAPER**                              | Use the fixed `rpc_auto.py` (tie‑break logic); or switch to `rpc_mit.py`. |

<br>


## AUTHORS

Darius Giannoli and Gabriel Taïeb  
