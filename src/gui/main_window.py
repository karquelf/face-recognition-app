from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QWidget
)

from src.gui.analyze_widget import AnalyzeWidget
from src.gui.cam_widget import CamWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("KSpy")

        self.analyzeWidget = AnalyzeWidget()
        self.CamWidget = CamWidget()

        self.hLayout = QHBoxLayout()
        self.hLayout.addWidget(self.CamWidget)
        self.hLayout.addWidget(self.analyzeWidget)

        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.hLayout)

        self.setCentralWidget(self.centralWidget)
