# app.py
import cv2
from fer import FER
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)
detector = FER()
cap = cv2.VideoCapture(0)   # Single camera instance

emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "🙂"
}

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        result = detector.detect_emotions(frame)
        for face in result:
            (x, y, w, h) = face["box"]
            emotion, score = detector.top_emotion(frame)

            if emotion:
                emoji = emoji_map.get(emotion.lower(), "🙂")
                text = f"{emoji} {emotion} ({score*100:.1f}%)"
            else:
                text = "🙂 Neutral (0%)"

            cv2.rectangle(frame, (x, y, x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    success, frame = cap.read()
    if not success:
        return jsonify({"emotion": "neutral", "score": 0})
    emotion, score = detector.top_emotion(frame)
    if emotion is None:
        return jsonify({"emotion": "neutral", "score": 0})
    return jsonify({"emotion": emotion, "score": score})

if __name__ == "__main__":
    app.run(debug=True)
