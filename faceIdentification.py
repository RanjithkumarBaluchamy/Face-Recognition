import cv2
import numpy as np
from google.colab import files

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

# Path to the folder containing images
dataset_path = "/content/Dataset"
images, labels, names = load_images_from_folder(dataset_path)

people_name = "Names"
names.append(people_name)

# Initialize the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(images, np.array(labels))

# Function to perform face recognition
def recognize_face(image):
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

# Function to capture images from webcam
def capture_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face recognition on the frame
        recognized_frame = recognize_face(frame)

        # Display the recognized frame
        cv2.imshow('Face Recognition', recognized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function to capture images from the webcam
capture_webcam()
