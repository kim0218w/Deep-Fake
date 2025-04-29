from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class GResultPage(QWidget):
    def __init__(self, stacked_widget, image_path):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        label = QLabel("딥페이크 생성 완료!")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        image_label = QLabel()
        pixmap = QPixmap(self.image_path)
        image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        layout.addWidget(image_label)

        back_btn = QPushButton("메인으로")
        back_btn.clicked.connect(self.goBack)
        layout.addWidget(back_btn)

        self.setLayout(layout)

    def goBack(self):
        self.stacked_widget.setCurrentIndex(0)
