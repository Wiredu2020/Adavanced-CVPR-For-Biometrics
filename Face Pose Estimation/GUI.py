import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class FacePoseEstimationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.capture = None
        self.timer = QTimer(self)

        # Load the pre-trained face pose estimation model
        self.model = load_model('inceptionv3_face_pose_model.h5')

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Face Pose Estimation App")
        self.setGeometry(100, 100, 1000, 700)  

        # Main layout
        main_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()
        self.result_label = QLabel("Predicted Pose: ")
        self.result_label.setStyleSheet("font-size: 16px; color: black;")
        main_layout.addWidget(self.result_label)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label, stretch=7)  

        self.upload_image_button = QPushButton('Upload Image', self)
        self.upload_image_button.setStyleSheet("background-color: blue; color: white;")
        self.upload_image_button.clicked.connect(self.upload_image)
        bottom_layout.addWidget(self.upload_image_button)

        self.start_webcam_button = QPushButton('Real-time', self)
        self.start_webcam_button.setStyleSheet("background-color: blue; color: white;")
        self.start_webcam_button.clicked.connect(self.start_webcam)
        bottom_layout.addWidget(self.start_webcam_button)

        self.restart_button = QPushButton('Restart', self)
        self.restart_button.setStyleSheet("background-color: red; color: white;")
        self.restart_button.clicked.connect(self.restart_application)
        bottom_layout.addWidget(self.restart_button)

    
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

        self.timer.timeout.connect(self.process_webcam_frame)

    def upload_image(self):
      
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if image_path:
            self.process_image(image_path)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            self.result_label.setText("No face detected.")
            return

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))  

       
            face_array = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = np.expand_dims(face_array, axis=0)  
            face_array = face_array / 255.0  

            # Predict pose
            prediction = self.model.predict(face_array)

            # Extract pitch, yaw, and roll values from the prediction
            pitch, yaw, roll = prediction[0]
            
            # Get the description of the head pose
            pose_description = self.describe_head_pose(pitch, yaw, roll)

            # Update the predicted pose in the label (not on the image)
            self.result_label.setText(f"Predicted Pose: {pose_description}")

            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with face detection
        self.display_image(img)

    def describe_head_pose(self, pitch, yaw, roll):
        pose_description = []

        # Pitch description
        if pitch < -15:
            pose_description.append("looking down")
        elif pitch > 15:
            pose_description.append("looking up")

        # Yaw description
        if yaw < -15:
            pose_description.append("turned to the right")
        elif yaw > 15:
            pose_description.append("turned to the left")

        # Roll description
        if roll < -10:
            pose_description.append("tilted to the left")
        elif roll > 10:
            pose_description.append("tilted to the right")

        # Combine all
        if not pose_description:
            return "The head is facing straight."
        return " and ".join(pose_description).capitalize() + "."

    def display_image(self, image):
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = channel * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_webcam(self):
        self.stop_capture()
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.result_label.setText("Error: Cannot access webcam.")
            return

        self.timer.start(10)  

    def process_webcam_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if not ret:
                return

            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))

                # Preprocess the face for prediction
                face_array = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_array = np.expand_dims(face_array, axis=0)
                face_array = face_array / 255.0

                # Predict pose
                prediction = self.model.predict(face_array)

                # Extract pitch, yaw, and roll values
                pitch, yaw, roll = prediction[0]

                # Get the pose description
                pose_description = self.describe_head_pose(pitch, yaw, roll)

                # Update the predicted pose in the result label (on UI)
                self.result_label.setText(f"Predicted Pose: {pose_description}")

                # Draw bounding box for face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the webcam feed
            self.display_image(frame)

    def restart_application(self):
        self.stop_capture()            
        self.result_label.setText("Predicted Pose: ") 
        self.image_label.clear()        

    def stop_capture(self):
        if self.capture:
            self.capture.release()
        self.timer.stop()

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FacePoseEstimationApp()
    window.show()
    sys.exit(app.exec_())
