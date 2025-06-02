#!/usr/bin/env python3
"""
rps_cam_mit.py — Rock‑Paper‑Scissors caméra + Iocaine‑lite (corrigé)
"""

import argparse, sys, time, random
import cv2, numpy as np, mediapipe as mp, PySimpleGUI as sg

# -------------------- CLI
cli = argparse.ArgumentParser()
cli.add_argument("--index", "--camera", type=int, default=0)
cli.add_argument("--backend", choices=["any","avf"], default="any")
cli.add_argument("--width", type=int, default=1280)
cli.add_argument("--height", type=int, default=720)
args = cli.parse_args()
W, H = args.width, args.height
BACK = cv2.CAP_AVFOUNDATION if args.backend=="avf" else cv2.CAP_ANY

# -------------------- caméra
cap = cv2.VideoCapture(args.index, BACK)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    sys.exit(f"Cannot open camera index {args.index}")
def grab():
    ok, f = cap.read()
    if not ok: raise RuntimeError("Camera grab failed")
    return f

# -------------------- MediaPipe
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
hands = mp_hands.Hands(False, 1, 0, 0.6, 0.5)
def fingers_up(lm):
    p = lm.landmark
    n = int(abs(p[4].x-p[2].x)>.04)
    n += sum(p[t].y<p[pip].y for t,pip in zip([8,12,16,20],[6,10,14,18]))
    return n
LABELS={0:"ROCK",2:"SCISSORS",5:"PAPER"}
def classify(n): return LABELS.get(n,"UNKNOWN")

# -------------------- Iocaine‑lite
MOVES=("R","P","S"); BEATS={"R":"P","P":"S","S":"R"}; LOSES={v:k for k,v in BEATS.items()}
IDX={"R":0,"P":1,"S":2}
def score(me,opp): return 0 if me==opp else 1 if BEATS[opp]==me else -1
class Freq:   predict=lambda s,h: max(MOVES,key=h.count) if h else random.choice(MOVES)
class Last:   predict=lambda s,h: h[-1] if h else random.choice(MOVES)
class Mark1:
    def __init__(s): s.t=np.ones((3,3),int)
    def predict(s,h):
        if len(h)>=2: s.t[IDX[h[-2]],IDX[h[-1]]]+=1
        return MOVES[np.argmax(s.t[IDX[h[-1]]])] if h else random.choice(MOVES)
class Meta:
    def __init__(s,b,lvl,lag):
        s.b,b = b,lvl; s.lvl=lvl; s.lag=lag; s.pred_prev=random.choice(MOVES)
        s.score=0.0; s.last=random.choice(MOVES)
    def choose(s,h):
        pred = s.pred_prev if s.lag else s.b.predict(h)
        if s.lag: s.pred_prev = s.b.predict(h)
        move = BEATS[pred] if s.lvl==0 else pred if s.lvl==1 else LOSES[pred]
        s.last=move; return move
class Iocaine:
    def __init__(s):
        bases=[Freq(),Last(),Mark1()]
        s.metas=[Meta(b,l,lag) for b in bases for l in (0,1,2) for lag in (0,1)]
        s.h=""
    def feed(s,opp):
        s.h+=opp
        for m in s.metas: m.score=0.97*m.score+score(m.last,opp)
    def next(s): return max(s.metas,key=lambda m:m.score).choose(s.h)
ioc=Iocaine(); MAP={"ROCK":"R","PAPER":"P","SCISSORS":"S"}

# -------------------- GUI
sg.theme("DarkBlue3")
layout=[[sg.Image(key="-IMG-",size=(W,H))],
        [sg.Text("State:"),sg.Text("WAIT",key="-STATE-"),
         sg.Text("  Coach:"),sg.Text("",key="-AI-",text_color="yellow")],
        [sg.Button("Start round",key="-START-",button_color=("white","green")),
         sg.Button("Quit",button_color=("white","firebrick3"))]]
win=sg.Window("RPS Iocaine Coach",layout,finalize=True,resizable=True); win.maximize()

# -------------------- FSM vars
WAIT, COUNTDOWN, SHOW = range(3)
state, t0, t_show = WAIT, 0, 0
COUNT_STR = ["3","2","1"]
detected, advice = "NO HAND","..."
prev,fps=time.time(),0.0

# -------------------- loop
try:
    while True:
        frame=cv2.flip(grab(),1); now=time.time()
        if state==COUNTDOWN:
            n=int(now-t0)
            if n<3:
                cv2.putText(frame,COUNT_STR[n],(W//2-40,H//2),
                            cv2.FONT_HERSHEY_DUPLEX,4,(0,215,255),6)
            else:
                res=hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    lm=res.multi_hand_landmarks[0]
                    detected=classify(fingers_up(lm))
                    mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)
                if detected in MAP: ioc.feed(MAP[detected])
                advice=ioc.next(); state,t_show=SHOW,now
        elif state==SHOW:
            cv2.putText(frame,f"Play: {advice}",(W//2-160,H//2),
                        cv2.FONT_HERSHEY_DUPLEX,2.5,(0,255,0),5)
            if now-t_show>2: state=WAIT

        fps=0.9*fps+0.1*(1/(now-prev)); prev=now
        cv2.putText(frame,f"{detected} {fps:4.1f} FPS",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        win["-IMG-"].update(data=cv2.imencode(".png",frame)[1].tobytes())
        win["-STATE-"].update(["WAIT","COUNT","SHOW"][state])
        win["-AI-"].update(advice)

        ev,_=win.read(timeout=1)
        if ev in (sg.WIN_CLOSED,"Quit"): break
        if ev in ("-START-","space") and state==WAIT:
            state,t0=COUNTDOWN,now; detected,advice="NO HAND","..."
finally:
    win.close(); hands.close(); cap.release()