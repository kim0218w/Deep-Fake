from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from G_ResultPage import GResultPage
from utils.fake_generator import fake_generate  # 딥페이크 모델 함수 불러오기
import os


def uploadImage(self):
    file_name, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg)")
    if file_name:
        output_path = os.path.join("generated_result.jpg")
        generate_deepfake(file_name, output_path)  # 생성 모델 호출

        # 결과 페이지에 output_path 전달
        result_page = GResultPage(self.stacked_widget, output_path)
        self.stacked_widget.addWidget(result_page)
        self.stacked_widget.setCurrentWidget(result_page)
