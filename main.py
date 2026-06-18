# app.py
import cv2
from fer import FER

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = FER()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)

    # Draw results on frame
    for face in result:
        (x, y, w, h) = face["box"]
        emotion, score = detector.top_emotion(frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Sentiment Analysis", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()