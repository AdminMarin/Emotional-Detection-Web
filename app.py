from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# --- LOAD MODELS ---
# Why: We load these once at the start. If we loaded them inside the loop, 
# the computer would crash trying to reload the AI 30 times a second.
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

# Define the emotions (Must match the order you trained them in!)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- THE CAMERA LOGIC ---
def generate_frames():
    camera = cv2.VideoCapture(0) # 0 = Default Laptop Webcam

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 1. Convert to Grayscale
            # Why: AI models are usually trained on gray images (lighter, faster).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Detect Faces
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            # 3. Process Each Face Found
            for (x, y, w, h) in faces:
                # Draw a box around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                # Crop just the face area
                roi_gray = gray[y:y+h, x:x+w]
                
                # Resize to 48x48 (Standard for emotion models like FER2013)
                # Why: Your model expects a specific input size. 
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                # Normalize the image data (0 to 1 scale)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0) # Make it a batch of 1

                    # Predict the emotion
                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    
                    # specific formatting for the text
                    label_position = (x, y - 10)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 4. Encode the frame
            # Why: Browsers can't display raw numpy arrays. We convert it to JPG bytes.
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # 5. Stream it
            # Why: 'yield' sends data piece by piece, creating a video stream.
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- THE WEBSITE ROUTES ---

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    # In a real app, you would check username/password here.
    return render_template('dashboard.html')

@app.route('/detect_emotion')
def detect_emotion():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    # This route is the "source" for the image tag in HTML
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)