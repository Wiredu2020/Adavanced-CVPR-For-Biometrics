import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from deepface import DeepFace  # Make sure DeepFace is installed


class EmotionEstimationApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.capture = None
        self.timer = QTimer(self)

        # Initialize OpenCV's face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Set up the UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Emotion Estimation App")
        self.setGeometry(100, 100, 1000, 700)  # Larger window size

        # Main layout
        main_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        # Label to display result
        self.result_label = QLabel("Predicted Emotion: ")
        self.result_label.setStyleSheet("font-size: 18px; color: black;")
        main_layout.addWidget(self.result_label)

        # Image Label for displaying video feed or image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label, stretch=7)  # Larger stretch factor for main display

        # Button: Upload Image
        self.upload_image_button = QPushButton('Upload Image', self)
        self.upload_image_button.setStyleSheet("background-color: blue; color: white;")
        self.upload_image_button.clicked.connect(self.upload_image)
        bottom_layout.addWidget(self.upload_image_button)

        # Button: Real-time Processing
        self.start_webcam_button = QPushButton('Real-time', self)
        self.start_webcam_button.setStyleSheet("background-color: blue; color: white;")
        self.start_webcam_button.clicked.connect(self.start_webcam)
        bottom_layout.addWidget(self.start_webcam_button)

        # Button: Restart
        self.restart_button = QPushButton('Restart', self)
        self.restart_button.setStyleSheet("background-color: red; color: white;")
        self.restart_button.clicked.connect(self.restart_application)
        bottom_layout.addWidget(self.restart_button)

        # Add bottom layout
        main_layout.addLayout(bottom_layout)

        # Set main layout for the window
        self.setLayout(main_layout)

        # Timer for webcam stream
        self.timer.timeout.connect(self.process_webcam_frame)

    def upload_image(self):
        # Open a file dialog to select an image
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if image_path:
            self.process_image(image_path)

    def process_image(self, image_path):
        # Process the uploaded image for face detection and emotion estimation
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image using Haar cascades
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            self.result_label.setText("No face detected.")
            return

        # Initialize the predicted_emotion to Unknown
        predicted_emotion = "Unknown"

        for (x, y, w, h) in faces:
            # Extract the face for emotion prediction
            face = img[y:y+h, x:x+w]

            try:
                # Use DeepFace to analyze emotions within the detected face region
                analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                predicted_emotion = analysis[0]['dominant_emotion']
            except Exception as e:
                # If there is an error in emotion prediction, display error message
                predicted_emotion = "Error: " + str(e)

            # Draw bounding box around the face and display predicted emotion
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Update the UI with the predicted emotion
        self.result_label.setText(f"Predicted Emotion: {predicted_emotion}")

        # Display the image with face detection and emotion prediction
        self.display_image(img)

    def display_image(self, image):
        # Convert BGR to RGB for displaying in QLabel
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image into a QPixmap
        height, width, channel = img_rgb.shape
        bytes_per_line = channel * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Dynamically scale the pixmap
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_webcam(self):
        self.stop_capture()
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.result_label.setText("Error: Cannot access webcam.")
            return

        self.timer.start(30)  # Process frame every 30 ms

    def process_webcam_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if not ret:
                return

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            # Initialize the predicted_emotion variable in case no face is detected
            predicted_emotion = "No Face detected, sit up right"

            for (x, y, w, h) in faces:
                # Extract the face for emotion prediction
                face = frame[y:y+h, x:x+w]

                try:
                    # Use DeepFace to analyze emotions within the detected face region
                    analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                    predicted_emotion = analysis[0]['dominant_emotion']
                except Exception as e:
                    # If there is an error in emotion prediction, display error message
                    predicted_emotion = "Error: " + str(e)

                # Draw bounding box around the face and display predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Update the UI with the predicted emotion
            self.result_label.setText(f"Predicted Emotion: {predicted_emotion}")

            # Display the webcam feed with face detection and emotion prediction
            self.display_image(frame)

    def restart_application(self):
        self.stop_capture()
        self.image_label.clear()
        self.result_label.setText("Predicted Emotion: ")

    def stop_capture(self):
        if self.capture:
            self.capture.release()
        self.timer.stop()

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionEstimationApp()
    window.show()
    sys.exit(app.exec_())
