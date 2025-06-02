#!/usr/bin/env python3
"""
Version simplifiée de Rock-Paper-Scissors avec débogage amélioré
"""

import argparse, sys, time
import cv2, numpy as np, mediapipe as mp

# -------------------- CLI --------------------
ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--width", type=int, default=640)  # Résolution réduite par défaut
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--confidence", type=float, default=0.5) # Seuil de confiance pour la détection
args = ap.parse_args()
W, H = args.width, args.height

# -------------------- Camera -----------------
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    sys.exit("Cannot open camera")

# Afficher les propriétés de la caméra (débogage)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Caméra initialisée : {actual_width}x{actual_height}")

def grab():
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera grab failed")
    return frame

# -------------------- MediaPipe --------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# Configurer MediaPipe avec un seuil de confiance ajusté
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.confidence,  # Réduit pour plus de sensibilité
    min_tracking_confidence=args.confidence
)

def fingers_up(lm_obj):
    pts = lm_obj.landmark
    # Afficher les coordonnées des points clés pour le débogage
    print(f"Thumb: ({pts[4].x:.3f}, {pts[4].y:.3f}) Base: ({pts[2].x:.3f}, {pts[2].y:.3f})")
    print(f"Index: ({pts[8].x:.3f}, {pts[8].y:.3f}) Middle: ({pts[12].x:.3f}, {pts[12].y:.3f})")
    
    # Calculer les doigts levés
    # Pouce: vérifier si le pouce est écarté horizontalement
    thumb_up = abs(pts[4].x - pts[2].x) > 0.04
    
    # Autres doigts: vérifier si les bouts des doigts sont plus hauts que les articulations
    other_fingers_up = [pts[i].y < pts[j].y for i, j in zip([8, 12, 16, 20], [6, 10, 14, 18])]
    
    # Afficher l'état de chaque doigt
    print(f"Doigts levés: Pouce={thumb_up}, Index={other_fingers_up[0]}, Majeur={other_fingers_up[1]}, Annulaire={other_fingers_up[2]}, Auriculaire={other_fingers_up[3]}")
    
    # Calcul final
    n = int(thumb_up) + sum(other_fingers_up)
    return n

def classify(n):
    result = {0:"ROCK", 2:"SCISSORS", 5:"PAPER"}.get(n, "UNKNOWN")
    print(f"Nombre de doigts détectés: {n} → {result}")
    return result

# -------------------- Markov AI --------------
idx = {"ROCK":0, "PAPER":1, "SCISSORS":2}
beats = {0:1, 1:2, 2:0}
count = np.ones((3,3), dtype=int)
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

# -------------------- Modes -----------------
MODE_WAIT = "WAIT"
MODE_COUNT = "COUNTDOWN"
MODE_SHOW = "SHOW_RESULT"
mode = MODE_WAIT
count_start = 0
result_time = 0
label = "NO HAND"
ai_move = "..."

prev, fps = time.time(), 0.0
COUNT_NUMBERS = ["3", "2", "1"]

print("\nRPS Terminal Version - Debug Mode")
print("--------------------------------")
print("Appuyez sur 's' pour démarrer le compte à rebours")
print("Appuyez sur 'd' pour détecter directement (sans compte à rebours)")
print("Appuyez sur 'q' pour quitter")
print("--------------------------------")

try:
    while True:
        frame = cv2.flip(grab(), 1)
        
        # Conversion en RGB pour MediaPipe (important)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection continue pour le mode debug
        if mode != MODE_COUNT:  # Ne pas détecter pendant le compte à rebours 
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Afficher l'état de détection en continu
                    if mode == MODE_WAIT:
                        fingers = fingers_up(hand_landmarks)
                        tmp_label = classify(fingers)
                        cv2.putText(frame, f"Détection: {tmp_label}", (10, 60),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # ---------------------------------- MODE transitions
        now = time.time()
        if mode == MODE_COUNT:
            elapsed = now - count_start
            idx_num = int(elapsed)
            if idx_num < 3:
                num = COUNT_NUMBERS[idx_num]
                cv2.putText(frame, num, (W//2-40, H//2),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (0,215,255), 6)
                print(f"Compte à rebours: {num}")
            else:
                print("\nAnalyse de la main...")
                # Cette fois, forcer l'analyse avec un nouveau frame
                detection_frame = cv2.cvtColor(grab(), cv2.COLOR_BGR2RGB)
                res = hands.process(detection_frame)
                
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    print("Main détectée!")
                    fingers = fingers_up(lm)
                    label = classify(fingers)
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                else:
                    print("Aucune main détectée après le compte à rebours!")
                    label = "NO HAND"
                    
                ai_move = recommend(label)
                mode = MODE_SHOW
                result_time = now
                print(f"\nRésultat:")
                print(f"Votre geste: {label}")
                print(f"Suggestion IA: {ai_move}")
                print("--------------------------------")

        elif mode == MODE_SHOW:
            cv2.putText(frame, f"Play: {ai_move}", (W//2-140, H//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,255,0), 5)
            if now - result_time > 2:
                mode = MODE_WAIT

        # Overlay FPS & dernier geste reconnu
        fps = 0.9*fps + 0.1*(1/(now-prev))
        prev = now
        cv2.putText(frame, f"{label}  {fps:4.1f} FPS", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Affichage d'une grille de référence
        cv2.line(frame, (0, H//2), (W, H//2), (100, 100, 100), 1)
        cv2.line(frame, (W//2, 0), (W//2, H), (100, 100, 100), 1)

        # Affichage du cadre
        cv2.imshow('Rock-Paper-Scissors Terminal', frame)
        
        # Vérification des touches
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and mode == MODE_WAIT:
            mode = MODE_COUNT
            count_start = now
            label = "NO HAND"
            ai_move = "..."
            print("\nDémarrage du compte à rebours!")
        if key == ord('d') and mode == MODE_WAIT:
            print("\nDétection directe demandée...")
            detection_frame = cv2.cvtColor(grab(), cv2.COLOR_BGR2RGB)
            res = hands.process(detection_frame)
            
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                print("Main détectée!")
                fingers = fingers_up(lm)
                label = classify(fingers)
                ai_move = recommend(label)
                print(f"Votre geste: {label}")
                print(f"Suggestion IA: {ai_move}")
                print("--------------------------------")
            else:
                print("Aucune main détectée lors de la détection directe!")

finally:
    cv2.destroyAllWindows()
    hands.close()
    cap.release()