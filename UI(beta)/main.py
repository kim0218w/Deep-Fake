import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from mainPage import MainPage

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    stacked_widget = QStackedWidget()
    
    main_page = MainPage(stacked_widget)
    stacked_widget.addWidget(main_page)
    
    stacked_widget.setFixedSize(800, 600)
    stacked_widget.setWindowTitle("Deepfake")
    stacked_widget.show()

    sys.exit(app.exec_())
