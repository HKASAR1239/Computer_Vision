#!/usr/bin/env python3
"""
rps_cam_ready_auto.py
—————————
• Waits for the “ready” pose: 1 fist (0 fingers) ABOVE 1 open hand (5 fingers)
• When detected ≥ 3 consecutive frames → starts countdown 3‑2‑1
• Freezes frame after “1”, classifies ROCK/PAPER/SCISSORS
• Suggests counter‑move (Markov‑1 coach) for 2 s, then loops
"""

import argparse, sys, time, random
import cv2, numpy as np, mediapipe as mp, PySimpleGUI as sg

# ---------- CLI --------------------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--camera", type=int, default=0)
cli.add_argument("--width",  type=int, default=1280)
cli.add_argument("--height", type=int, default=720)
args = cli.parse_args()
W, H = args.width, args.height

# ---------- camera -----------------------------------------------------------
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    sys.exit("Cannot open camera")
grab = lambda: cap.read()[1]

# ---------- MediaPipe --------------------------------------------------------
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
hands = mp_hands.Hands(False, 2, 0, 0.6, 0.5)

def fingers_up(lm):
    p = lm.landmark
    n = int(abs(p[4].x - p[2].x) > .04)                # thumb
    n += sum(p[t].y < p[q].y for t, q in
             zip([8, 12, 16, 20], [6, 10, 14, 18]))
    return n

def ready_pose(lms):
    """Return True if two hands form fist‑over‑palm (0 vs 5 fingers)."""
    if len(lms) != 2:
        return False
    A, B = lms
    na, nb = fingers_up(A), fingers_up(B)
    ya = np.mean([pt.y for pt in A.landmark])
    yb = np.mean([pt.y for pt in B.landmark])
    top_n, bot_n = (na, nb) if ya < yb else (nb, na)
    return top_n == 0 and bot_n == 5

LABELS = {0: "ROCK", 2: "SCISSORS", 5: "PAPER"}
def classify(n): return LABELS.get(n, "UNKNOWN")

# ---------- Markov 1 coach ---------------------------------------------------
idx = {"ROCK": 0, "PAPER": 1, "SCISSORS": 2}
beats = {0: 1, 1: 2, 2: 0}
cnt = np.ones((3, 3), int); last = None
def recommend(cur):
    global last, cnt
    if cur in idx and last in idx:
        cnt[idx[last], idx[cur]] += 1
    last = cur if cur in idx else last
    if last is None:
        return random.choice(list(idx))
    row = cnt[idx[last]]; m = row.max()
    nxt = int(random.choice(np.flatnonzero(row == m)))
    return ["ROCK", "PAPER", "SCISSORS"][beats[nxt]]

# ---------- GUI --------------------------------------------------------------
sg.theme("DarkBlue3")
layout = [[sg.Image(key="-IMG-", size=(W, H))],
          [sg.Text("State:"), sg.Text("", key="-STATE-"),
           sg.Text("   Coach:"), sg.Text("", key="-AI-", text_color="yellow")],
          [sg.Text("Show fist over open hand to start a round",
                   text_color="grey"),
           sg.Button("Quit", button_color=("white", "firebrick3"))]]
win = sg.Window("RPS Ready‑Pose Coach", layout, finalize=True, resizable=True)
win.maximize()

# ---------- FSM --------------------------------------------------------------
WAIT, COUNT, SHOW = range(3)
state, t0 = WAIT, 0
COUNT_TXT = ["3", "2", "1"]
label, ai = "NO HAND", "..."
consec_ready = 0
prev, fps = time.time(), 0.0

try:
    while True:
        frame = grab()
        if frame is None:                       # camera hiccup
            if win.read(timeout=1)[0] in (sg.WIN_CLOSED, "Quit"): break
            continue
        frame = cv2.flip(frame, 1)
        now = time.time()

        # ------------- detection (runs regardless of state)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ------------- WAIT -> COUNT trigger
        if state == WAIT:
            if res.multi_hand_landmarks and ready_pose(res.multi_hand_landmarks):
                consec_ready += 1
                cv2.putText(frame, "READY!", (W//2 - 80, H//2),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 215, 255), 4)
                if consec_ready >= 3:           # ≈100 ms stability
                    state, t0 = COUNT, now
                    label, ai = "NO HAND", "..."
                    consec_ready = 0
            else:
                consec_ready = 0

        # ------------- COUNTDOWN logic
        elif state == COUNT:
            n = int(now - t0)
            if n < 3:
                cv2.putText(frame, COUNT_TXT[n], (W//2 - 40, H//2),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (0, 215, 255), 6)
            else:
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    label = classify(fingers_up(lm))
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                ai = recommend(label)
                state, t0 = SHOW, now

        # ------------- SHOW result
        elif state == SHOW:
            cv2.putText(frame, f"Play: {ai}", (W//2 - 160, H//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 0), 5)
            if now - t0 > 2:
                state = WAIT

        # ------------- HUD
        fps = 0.9 * fps + 0.1 * (1 / (now - prev)); prev = now
        cv2.putText(frame, f"{label}  {fps:4.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ------------- GUI update
        win["-IMG-"].update(data=cv2.imencode(".png", frame)[1].tobytes())
        win["-STATE-"].update(["WAIT", "COUNT", "SHOW"][state])
        win["-AI-"].update(ai)

        if win.read(timeout=1)[0] in (sg.WIN_CLOSED, "Quit"):
            break
finally:
    win.close(); hands.close(); cap.release()
