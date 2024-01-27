import cv2
import concurrent.futures
from datetime import datetime

from src.face_recognition.cam_detector import detect_faces, identify_faces

from PySide6.QtCore import (
    Qt,
    QTimer
)

from PySide6.QtGui import (
    QImage,
    QPixmap
)

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)

DETECTION_FREQUENCY = 10
IDENTIFICATION_FREQUENCY = 60
MOVE_SPEED = 2

class CamWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.recording = False
        self.frame_count = 0
        self.last_time = None
        self.delta = None
        self.face_locations = []
        self.bounding_boxes = []
        self.names = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        self.setFixedSize(640, 480)

        self.label = QLabel("Camera feed")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)

        self.startButton = QPushButton("Start recording")
        self.startButton.setFixedWidth(100)
        self.startButton.clicked.connect(self.startRecording)

        self.stopButton = QPushButton("Stop recording")
        self.stopButton.setFixedWidth(100)
        self.stopButton.clicked.connect(self.stopRecording)
        self.stopButton.hide()

        self.videoCapture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)

        self.hLayout = QHBoxLayout()
        self.hLayout.addWidget(self.startButton)
        self.hLayout.addWidget(self.stopButton)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addLayout(self.hLayout)

        self.setLayout(self.layout)

    def startRecording(self):
        self.recording = True
        self.startButton.hide()
        self.stopButton.show()
        self.timer.start(30) # Set the interval in milliseconds (e.g., 30ms for 30 frames per second)

    def stopRecording(self):
        self.recording = False
        self.stopButton.hide()
        self.startButton.show()
        self.timer.stop()
        self.label.setText("Camera feed")

    def updateFaceLocations(self, frame):
        self.face_locations = detect_faces(frame)

    def updateNames(self, frame):
        self.names = identify_faces(frame, self.face_locations)

    def updateFrameTime(self):
        if self.last_time:
            self.delta = datetime.now() - self.last_time
        self.last_time = datetime.now()

    def updateFrame(self):
        self.updateFrameTime()

        ret, frame = self.videoCapture.read()
        if ret:
            # Convert OpenCV image to QImage
            height, width, _channel = frame.shape
            bytes_per_line = 3 * width

            self.frame_count += 1
            if self.frame_count % DETECTION_FREQUENCY == 0:
                # Find all the faces and face enqcodings in the frame of video
                self.executor.submit(self.updateFaceLocations, frame)

            if self.frame_count % IDENTIFICATION_FREQUENCY == 0:
                self.executor.submit(self.updateNames, frame)

            new_bounding_boxes = []
            for index, face_location in enumerate(self.face_locations):
                top, right, bottom, left = self.calculateBoxPosition(face_location, index)
                new_bounding_boxes.append((top, right, bottom, left))
            self.bounding_boxes = new_bounding_boxes

            for bounding_box in self.bounding_boxes:
                top, right, bottom, left = bounding_box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            for index, name in enumerate(self.names):
                cv2.putText(frame, name, (40, 80 + index * 40), font, 1.5, (0, 0, 255), 4)

            qImage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Convert QImage to QPixmap for displaying in QLabel
            pixmap = QPixmap.fromImage(qImage)
            pixmap.setDevicePixelRatio(1)

            # Update the QLabel with the new frame
            self.label.setPixmap(pixmap)

    def calculateBoxPosition(self, face_location, index):
        try:
            current_top, current_right, current_bottom, current_left = self.bounding_boxes[index]
            goal_top, goal_right, goal_bottom, goal_left = face_location

            top = current_top + (goal_top - current_top) * self.delta.total_seconds() * MOVE_SPEED
            right = current_right + (goal_right - current_right) * self.delta.total_seconds() * MOVE_SPEED
            bottom = current_bottom + (goal_bottom - current_bottom) * self.delta.total_seconds() * MOVE_SPEED
            left = current_left + (goal_left - current_left) * self.delta.total_seconds() * MOVE_SPEED

            return (int(top), int(right), int(bottom), int(left))

        except IndexError:
            return face_location
