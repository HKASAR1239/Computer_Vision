#!/usr/bin/env python3
"""
Rock‑Paper‑Scissors
-------------------------------------
• Countdown (Space or Start)
• Detects (Rock/Paper/Scissors) just after '1'
• Recommand the best move (first order Markov Chain)
• loop : wait -> countdown -> resukt -> wait...
"""

import argparse, sys, time
import cv2, numpy as np, mediapipe as mp, PySimpleGUI as sg

# -------------------- CLI --------------------
ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--width",  type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
args = ap.parse_args()
W, H = args.width, args.height

# -------------------- Camera -----------------
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    sys.exit("Cannot open camera")

def grab():
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera grab failed")
    return frame

# -------------------- MediaPipe --------------
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
hands = mp_hands.Hands(False, 1, 0, 0.6, 0.5)

def fingers_up(lm_obj):
    pts = lm_obj.landmark
    tips, pips = [4,8,12,16,20], [2,6,10,14,18]
    n = int(abs(pts[4].x - pts[2].x) > .04)
    n += sum(pts[t].y < pts[p].y for t,p in zip(tips[1:], pips[1:]))
    return n

def classify(n):
    return {0:"ROCK", 2:"SCISSORS", 5:"PAPER"}.get(n, "UNKNOWN")

# -------------------- Markov AI --------------
idx = {"ROCK":0, "PAPER":1, "SCISSORS":2}
beats = {0:1, 1:2, 2:0}                # rock→paper, paper→scissors, scissors→rock
count = np.ones((3,3), dtype=int)      # Laplace smoothing
last_state = None
def recommend(cur):
    global last_state, count
    if cur in idx and last_state in idx:
        count[idx[last_state], idx[cur]] += 1
    last_state = cur if cur in idx else last_state
    if last_state is None:
        return "..."
    next_probable = np.argmax(count[idx[last_state]])
    return ["ROCK","PAPER","SCISSORS"][beats[next_probable]]

# -------------------- GUI --------------------
sg.theme("DarkBlue3")
layout = [
    [sg.Image(key="-IMG-", size=(W, H))],
    [sg.Text("Round state:"), sg.Text("WAIT", key="-STATE-"),
     sg.Text("   Your move:", pad=((20,0),0)),
     sg.Text("", key="-AI-", text_color="yellow")],
    [sg.Button("Start round", key="-START-", button_color=("white","green")),
     sg.Button("Quit", button_color=("white","firebrick3"))]
]
win = sg.Window("RPS Countdown Coach", layout, finalize=True, resizable=True)
win.maximize()

# -------------------- FSM states -------------
MODE_WAIT   = "WAIT"
MODE_COUNT  = "COUNTDOWN"
MODE_SHOW   = "SHOW_RESULT"
mode = MODE_WAIT
count_start = 0
result_time = 0
label = "NO HAND"
ai_move = "..."

prev, fps = time.time(), 0.0
COUNT_NUMBERS = ["3", "2", "1"]

try:
    while True:
        frame = cv2.flip(grab(), 1)

        # ---------------------------------- MODE transitions
        now = time.time()
        if mode == MODE_COUNT:
            elapsed = now - count_start
            idx_num = int(elapsed)             # 0 -> '3', 1 -> '2', 2 -> '1'
            if idx_num < 3:
                num = COUNT_NUMBERS[idx_num]
                cv2.putText(frame, num, (W//2-40, H//2),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (0,215,255), 6)
            else:
                # freeze this frame for detection
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    label = classify(fingers_up(lm))
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                ai_move = recommend(label)
                mode = MODE_SHOW
                result_time = now

        elif mode == MODE_SHOW:
            cv2.putText(frame, f"Play: {ai_move}", (W//2-140, H//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,255,0), 5)
            if now - result_time > 2:          # show result 2 s
                mode = MODE_WAIT

        # overlay FPS & last recognized gesture
        fps = 0.9*fps + 0.1*(1/(now-prev)); prev = now
        cv2.putText(frame, f"{label}  {fps:4.1f} FPS", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # ---------------------------------- GUI update
        win["-IMG-"].update(data=cv2.imencode(".png", frame)[1].tobytes())
        win["-STATE-"].update(mode)
        win["-AI-"].update(ai_move)

        ev, _ = win.read(timeout=1)
        if ev in (sg.WIN_CLOSED, "Quit"):
            break
        if ev in ("-START-", "space") and mode == MODE_WAIT:
            mode = MODE_COUNT
            count_start = now
            label = "NO HAND"
            ai_move = "..."

finally:
    win.close(); hands.close(); cap.release()
