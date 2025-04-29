from utils.my_detector import detect_deepfake  # 네 탐지 모델 import
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from D_ResultPage import DResultPage

def uploadVideo(self):
    file_name, _ = QFileDialog.getOpenFileName(self, "비디오 선택", "", "Videos (*.mp4 *.avi *.mov)")
    if file_name:
        result = detect_deepfake(file_name)  # True / False / 확률 등 반환

        # 결과 페이지에 파일명과 탐지 결과 전달
        result_page = DResultPage(self.stacked_widget, file_name, result)
        self.stacked_widget.addWidget(result_page)
        self.stacked_widget.setCurrentWidget(result_page)
