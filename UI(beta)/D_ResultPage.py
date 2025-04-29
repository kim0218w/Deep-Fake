from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

class DResultPage(QWidget):
    def __init__(self, stacked_widget, video_path, detection_result):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.video_path = video_path
        self.detection_result = detection_result  # 탐지 결과 추가
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        label = QLabel("탐지 결과")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # 탐지 결과 표시
        result_text = "딥페이크입니다!" if self.detection_result else "정상 영상입니다!"
        result_label = QLabel(f"분석 대상: {self.video_path}\n결과: {result_text}")
        result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(result_label)

        back_btn = QPushButton("메인으로")
        back_btn.clicked.connect(self.goBack)
        layout.addWidget(back_btn)

        self.setLayout(layout)

    def goBack(self):
        self.stacked_widget.setCurrentIndex(0)
