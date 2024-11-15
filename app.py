from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load your model
model = load_model('model_file_30epochs.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 3)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
