#  ASL Translator – Real-Time Sign Language Letter Recognition

This project is a **real-time American Sign Language (ASL) letter recognition web app**. It uses a webcam, computer vision, and hand landmark detection to identify ASL letters (A–Z, except J and Z) and displays them on a web interface along with a live video stream.

>  Currently supports static hand signs only. Signs involving motion (like J and Z) are not yet implemented.

---

##  Features

-  Real-time ASL letter detection using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
-  Live webcam video feed via OpenCV
-  Clean, responsive HTML interface served through Flask
-  Continual finger state updates and recognized letter stream
-  Public sharing via [ngrok](https://ngrok.com/)
-  Clear button to reset the current word

---

Tech Stack

- **Python 3**
- **OpenCV**
- **MediaPipe**
- **Flask** (Web backend)
- **Ngrok** (for external access)
- **HTML/CSS/JavaScript** (Frontend)

---

Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Keng-cryption/jetson-nano-AI.git
cd asl-translator
```

### 2. Install Required Python Packages

```bash
pip install opencv-python mediapipe flask flask-cors pyngrok
```

### 3. (Optional) Set Your Ngrok Authtoken

To avoid ngrok errors, authenticate once:

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken

---

Run the App

```bash
python app.py
```

After startup:

- You'll see a **public ngrok URL** in the terminal.
- Open that URL in your browser to access the live ASL translator web interface.
  
Web Interface

- **Live camera feed** showing hand detection
- **Current Word**: Letters translated in real time
- **Finger State**: Binary values for thumb/index/middle/ring/pinky
- **Clear**: Button to reset the current word

Recognized Letters

The app detects most letters based on finger positions:

- ✅ Supported: A–I, K–Y
- ❌ Not Supported: J, Z *(due to motion dependency)*


Troubleshooting

- **Black video screen?** Make sure your webcam is accessible and not used by another app.
- **Ngrok not working?** Check your internet connection and ngrok token.
- **Performance issues?** Reduce resolution or run on a faster device (like Jetson Nano or Pi 4).

