from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QFileDialog,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget
)

from PySide6.QtGui import (
    QPixmap
)

from PIL.ImageQt import ImageQt

from src.face_recognition.image_detector import recognize_faces

class AnalyzeWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.selectFileButton = QPushButton("Browse")
        self.selectFileButton.setFixedWidth(100)
        self.selectFileButton.clicked.connect(self.selectFile)

        self.fileSelectedLabel = QLabel("No File Selected")

        self.graphicsView = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setFixedSize(640, 640)
        self.graphicsView.setInteractive(False)

        self.pixmapItem = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmapItem)

        hLayout = QHBoxLayout()
        hLayout.addWidget(self.selectFileButton)
        hLayout.addWidget(self.fileSelectedLabel)

        vLayout = QVBoxLayout()
        vLayout.addLayout(hLayout)
        vLayout.addWidget(self.graphicsView)

        self.setLayout(vLayout)

    def wheelEvent(self, event):
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self.graphicsView.scale(factor, factor)
        event.accept()

    def calculateZoom(self):
        # Calculate the initial zoom factor to fit the entire image within the view
        view_rect = self.graphicsView.contentsRect()
        pixmap_rect = self.pixmapItem.pixmap().rect()

        initial_factor = min(view_rect.width() / pixmap_rect.width(), view_rect.height() / pixmap_rect.height())

        # Set the initial zoom factor
        self.graphicsView.resetTransform()
        self.graphicsView.scale(initial_factor, initial_factor)


    def selectFile(self):
        fileDialog = QFileDialog(self)
        fileDialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif);;Videos (*.mp4 *.avi)")
        fileDialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        fileDialog.setViewMode(QFileDialog.ViewMode.Detail)
        if fileDialog.exec():
            files = fileDialog.selectedFiles()
            if files:
                self.filePath = fileDialog.selectedFiles()[0]
                self.fileSelectedLabel.setText(self.filePath.split("/")[-1])
                self.analyze()

    def analyze(self):
        self.pixmapItem.setPixmap(QPixmap.fromImage(ImageQt(recognize_faces(self.filePath))))
        self.calculateZoom()

