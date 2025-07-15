import cv2
import mediapipe as mp
import time
import threading
from flask import Flask, jsonify, render_template_string, Response
from flask_cors import CORS
from pyngrok import ngrok

# Shared state
current_word = ""
finger_list = [0, 0, 0, 0, 0]
frame_for_stream = None
lock = threading.Lock()

# Flask app
app = Flask(__name__)
CORS(app)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ASL Translator</title>
    <style>
        /* Global styles */
        body {
            font-family: 'Open Sans', sans-serif;
            background-color: #1E2229; /* Dark teal */
            color: #fff;
            text-align: center;
            padding-top: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
        }

        /* Main content area */
        .main-content {
            background-color: rgba(0, 0, 0, 0.1); /* Light gray */
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
            transition: box-shadow 0.3s ease-out;
        }

        .main-content::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 50px; /* Visual flare */
            background-color: #1E7A54; /* Light teal */
            border-radius: 12px 12px 0 0;
        }

        .main-content::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 50px; /* Visual flare */
            background-color: #1E7A54; /* Light teal */
            border-radius: 0 0 12px 12px;
        }

        /* Title area */
        .title-area {
            text-align: center;
            margin-bottom: 20px;
        }

        h1 {
            font-size: 2.5em;
            font-weight: bold;
            color: #fff;
        }

        p, button {
            font-size: 1.3em;
            margin-top: 10px;
            color: #eee;
        }

        img {
            margin-top: 20px;
            border-radius: 12px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
        }

        /* Clear button */
        .clear-button {
            background-color: #1E7A54; /* Light teal */
            border: none;
            padding: 10px 20px;
            font-size: 1.5em;
            font-weight: bold;
            cursor: pointer;
        }

        .clear-button:hover {
            background-color: #2C3E50; /* Darker light teal */
        }
    </style>
</head>
<body>
    <div class="title-area">
        <h1>Live ASL Translation</h1>
        <button class="clear-button" onclick="clearWord()">Clear</button>
    </div>
    <div class="main-content">
        <p id="word">Loading...</p>
        <p id="fingers">Loading...</p>
        <img id="video" src="/video_feed" width="640" height="480">
    </div>

    <script>
        async function fetchData() {
            const res = await fetch('/status');
            const data = await res.json();
            document.getElementById("word").textContent = "Current Word: " + data.word;
            document.getElementById("fingers").textContent = "Finger State: " + data.fingers.join(", ");
        }

        async function clearWord() {
            await fetch('/clear');
            fetchData();
        }

        setInterval(fetchData, 1000);
        fetchData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    with lock:
        return jsonify(word=current_word, fingers=finger_list)

@app.route('/clear')
def clear():
    global current_word
    with lock:
        current_word = ""
    return jsonify(success=True)

@app.route('/video_feed')
def video_feed():
    def generate_video():
        global frame_for_stream
        while True:
            if frame_for_stream is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame_for_stream)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ASL logic
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

# Single thread for ASL + video feed
def asl_and_video_thread():
    global current_word, finger_list, frame_for_stream
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        last_letter = ""
        last_check = 0
        interval = 1.0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if time.time() - last_check >= interval:
                    last_check = time.time()
                    lm = result.multi_hand_landmarks[0].landmark
                    fingers = get_finger_states(lm)

                    flist = [
                        int(fingers['thumb']),
                        int(fingers['index']),
                        int(fingers['middle']),
                        int(fingers['ring']),
                        int(fingers['pinky']),
                    ]

                    with lock:
                        finger_list = flist
                        letter = classify_letter(fingers)
                        if letter and letter != last_letter:
                            current_word += letter
                            last_letter = letter
                        elif not letter:
                            last_letter = ""

            frame_for_stream = frame.copy()

    cap.release()

# Main entry
if __name__ == '__main__':
    threading.Thread(target=asl_and_video_thread, daemon=True).start()

    try:
        public_url = ngrok.connect(5000)
        print(f" * Public URL: {public_url}")
    except Exception as e:
        print(f"Ngrok failed: {e}")

    app.run(port=5000, threaded=True)
