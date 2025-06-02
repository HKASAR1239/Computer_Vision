import cv2
import numpy as np
import mediapipe as mp
import subprocess
import time
import sys

# Configuration
THUMB_UP_FRAMES = 8       
COOLDOWN_TIME = 3.0       
CONFIDENCE = 0.4          
CAMERA_INDEX = 0         

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE,
    min_tracking_confidence=CONFIDENCE
)

def is_thumb_up(landmarks):
    pts = landmarks.landmark 
    thumb_tip_y = pts[4].y
    index_tip_y = pts[8].y
    middle_tip_y = pts[12].y
    ring_tip_y = pts[16].y
    pinky_tip_y = pts[20].y
    
    thumb_highest = (thumb_tip_y < index_tip_y - 0.03 and
                    thumb_tip_y < middle_tip_y - 0.03 and
                    thumb_tip_y < ring_tip_y - 0.03 and
                    thumb_tip_y < pinky_tip_y - 0.03)
    
    thumb_base_y = pts[2].y
    thumb_extended = thumb_tip_y < thumb_base_y - 0.05
    
    if thumb_highest and thumb_extended:
        print(f"✓ Pouce levé détecté! (Y: {thumb_tip_y:.3f})")
    
    return thumb_highest and thumb_extended

def draw_status(frame, status_text, color=(0, 255, 0)):
    """Affiche le statut sur l'image."""
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def execute_selection():
    """Exécute selection.py et gère les erreurs."""
    try:
        print("\n🎯 Exécution de selection.py...")
        result = subprocess.run(
            ["python", "selection.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Coup joué avec succès!")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"❌ Erreur lors de l'exécution (code: {result.returncode})")
            if result.stderr:
                print(f"Erreur: {result.stderr}")
                
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏱️ Timeout - selection.py a pris trop de temps")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    # Configuration de la caméra
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("♟️ Chess Controller - Détection du pouce levé")
    print("=" * 50)
    print("Levez le pouce pour jouer un coup")
    print("Commandes: 'q' = quitter, 's' = jouer (debug)")
    print("=" * 50)
    
    # Variables d'état
    thumb_up_counter = 0
    last_play_time = 0
    waiting_cooldown = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur de lecture caméra")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            current_time = time.time()
            status = "En attente du pouce levé..."
            color = (0, 255, 0)  
            
            if waiting_cooldown:
                remaining = COOLDOWN_TIME - (current_time - last_play_time)
                if remaining > 0:
                    status = f"Cooldown: {remaining:.1f}s"
                    color = (0, 165, 255)  
                else:
                    waiting_cooldown = False
                    thumb_up_counter = 0
            
            if results.multi_hand_landmarks and not waiting_cooldown:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    if is_thumb_up(hand_landmarks):
                        thumb_up_counter += 1
                        status = f"Pouce détecté: {thumb_up_counter}/{THUMB_UP_FRAMES}"
                        color = (0, 255, 255)  # Jaune
                        
                        if thumb_up_counter >= THUMB_UP_FRAMES:
                            status = "Coup en cours..."
                            color = (0, 0, 255)  # Rouge
                            draw_status(frame, status, color)
                            cv2.imshow("Chess Controller", frame)
                            cv2.waitKey(1)  # Forcer l'affichage
                            
                            # Exécuter selection.py
                            success = execute_selection()
                            
                            # Réinitialiser les compteurs
                            thumb_up_counter = 0
                            last_play_time = current_time
                            waiting_cooldown = True
                            
                            if not success:
                                print("⚠️ Le coup a peut-être échoué, continuez...")
                    else:
                        thumb_up_counter = max(0, thumb_up_counter - 1)
            else:
                thumb_up_counter = max(0, thumb_up_counter - 1)
            
            draw_status(frame, status, color)
            
            cv2.putText(frame, "'q' = quitter, 's' = jouer", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)
            cv2.line(frame, (0, h//2), (w, h//2), (100, 100, 100), 1)
            
            cv2.imshow("Chess Controller", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Arrêt du contrôleur d'échecs")
                break
            elif key == ord('s') and not waiting_cooldown:
                print("\n[DEBUG] Exécution forcée de selection.py")
                execute_selection()
                last_play_time = current_time
                waiting_cooldown = True
                
    except KeyboardInterrupt:
        print("\n⚠️ Interruption par l'utilisateur")
    finally:
        # Nettoyage
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("✅ Ressources libérées")

if __name__ == "__main__":
    main()