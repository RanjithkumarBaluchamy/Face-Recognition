###face identification by accessing webcam through javascript


from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

import cv2
import numpy as np
import os

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    names = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            label = len(names)  # Assign a unique label for each person
            labels.append(label)
            names.append(filename.split('.')[0])  # Extract name from the file name
    return images, labels, names

# Function to capture images from webcam using JavaScript
def capture_webcam_js():
    js = Javascript('''
        async function captureWebcam() {
            const div = document.createElement('div');
            document.body.appendChild(div);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for the video to settle before capturing the frame
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Capture the frame and send it to Python
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            div.appendChild(canvas);
            const img = canvas.toDataURL('image/jpeg');
            div.remove();
            return img;
        }
    ''')
    display(js)

# Function to convert JavaScript image data to OpenCV format
def js_to_image(js_reply):
    image_bytes = b64decode(js_reply.split(',')[1])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return frame

# Function to perform face recognition
def recognize_face(image, names):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi_gray)
        if confidence < 100:
            name = names[label]
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return image

# Path to the folder containing images
dataset_path = "/content/Dataset"
images, labels, names = load_images_from_folder(dataset_path)

# Add your name to the names list
people_name_name = "Your Name"
names.append(people_name)

# Initialize the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(images, np.array(labels))

# Call the JavaScript function to capture webcam images
capture_webcam_js()

# Get the captured image data from JavaScript
js_reply = eval_js('captureWebcam()')

# Convert JavaScript image data to OpenCV format
frame = js_to_image(js_reply)

# Perform face recognition on the frame
recognized_frame = recognize_face(frame, names)

# Display the recognized frame
cv2.imshow('Face Recognition', recognized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
