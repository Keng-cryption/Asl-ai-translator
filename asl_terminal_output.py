import cv2
import mediapipe as mp
import time
import sys
import signal
import shutil

def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def get_finger_states(lm):
    return {
        'thumb': lm[4].x < lm[3].x,
        'index': finger_up(lm, 8, 6),
        'middle': finger_up(lm, 12, 10),
        'ring': finger_up(lm, 16, 14),
        'pinky': finger_up(lm, 20, 18)
    }

def classify_letter(f):
    thumb = f['thumb']
    fingers = [int(f['index']), int(f['middle']), int(f['ring']), int(f['pinky'])]
    for letter, pattern in LETTER_SIGNS.items():
        if thumb == pattern['thumb'] and fingers == pattern['fingers']:
            return letter
    return None

LETTER_SIGNS = {
    'A':  { 'thumb': True,  'fingers': [0, 0, 0, 0] },
    'B':  { 'thumb': False, 'fingers': [1, 1, 1, 1] },
    'C':  { 'thumb': True,  'fingers': [1, 0, 0, 1] },
    'D':  { 'thumb': False, 'fingers': [1, 0, 0, 0] },
    'E':  { 'thumb': False, 'fingers': [0, 0, 0, 0] },
    'F':  { 'thumb': True,  'fingers': [0, 1, 1, 1] },
    'G':  { 'thumb': True,  'fingers': [0, 1, 1, 0] },
    'H':  { 'thumb': False, 'fingers': [1, 1, 0, 0] },
    'I':  { 'thumb': False, 'fingers': [0, 0, 0, 1] },
    'K':  { 'thumb': False, 'fingers': [1, 0, 1, 1] },
    'L':  { 'thumb': True,  'fingers': [1, 0, 0, 0] },
    'M':  { 'thumb': False, 'fingers': [0, 1, 1, 0] },
    'N':  { 'thumb': True,  'fingers': [1, 1, 1, 0] },
    'O':  { 'thumb': True,  'fingers': [0, 0, 1, 1] },
    'P':  { 'thumb': True,  'fingers': [1, 0, 1, 1] },
    'Q':  { 'thumb': True,  'fingers': [0, 1, 0, 0] },
    'R':  { 'thumb': False, 'fingers': [1, 1, 0, 1] },
    'S':  { 'thumb': False, 'fingers': [0, 1, 0, 1] },
    'T':  { 'thumb': False, 'fingers': [0, 0, 1, 0] },
    'U':  { 'thumb': True,  'fingers': [1, 1, 0, 1] },
    'V':  { 'thumb': True,  'fingers': [1, 1, 0, 0] },
    'W':  { 'thumb': False, 'fingers': [1, 1, 1, 0] },
    'Y':  { 'thumb': True,  'fingers': [0, 0, 0, 1] },
    ' ':  { 'thumb': True,  'fingers': [1, 1, 1, 1] },
}

def main():
    print("ASL + Finger State Detection (Ctrl+C to quit)")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        current_word = ""
        last_letter = ""
        interval = 1
        last_check = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            now = time.time()
            if now - last_check >= interval:
                last_check = now

                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0].landmark
                    fingers = get_finger_states(lm)

                    finger_list = [
                        int(fingers['thumb']),
                        int(fingers['index']),
                        int(fingers['middle']),
                        int(fingers['ring']),
                        int(fingers['pinky']),
                    ]

                    letter = classify_letter(fingers)

                    if letter and letter != last_letter:
                        current_word += letter
                        last_letter = letter
                    elif not letter:
                        last_letter = ""

                    line1 = f"Current word: {current_word}"
                    line2 = f"Hand: {finger_list}"

                    width = shutil.get_terminal_size().columns
                    line1 = line1.ljust(width)
                    line2 = line2.ljust(width)
                    sys.stdout.write("\033[F\033[K")  
                    sys.stdout.write(f"{line1}\n{line2}")
                    sys.stdout.flush()

    cap.release()
    print("\n Camera released. Goodbye.")

if __name__ == "__main__":
    main()

