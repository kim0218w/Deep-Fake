from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QFont
from GenerationPage import GenerationPage
from DetectionPage import DetectionPage

class MainPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        label = QLabel("Deepfake")
        label.setFont(QFont("Arial", 24))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        btn_gen = QPushButton("Generation")
        btn_gen.clicked.connect(self.goToGeneration)
        layout.addWidget(btn_gen)

        btn_det = QPushButton("Detection")
        btn_det.clicked.connect(self.goToDetection)
        layout.addWidget(btn_det)

        self.setLayout(layout)

    def goToGeneration(self):
        page = GenerationPage(self.stacked_widget)
        self.stacked_widget.addWidget(page)
        self.stacked_widget.setCurrentWidget(page)

    def goToDetection(self):
        page = DetectionPage(self.stacked_widget)
        self.stacked_widget.addWidget(page)
        self.stacked_widget.setCurrentWidget(page)
