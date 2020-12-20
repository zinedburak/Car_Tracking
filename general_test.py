import sys

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow

from functionalities import *


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(os.path.join(os.path.dirname(__file__), "form6.ui"), self)
        self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.button_OpenFile.clicked.connect(lambda: openFile(self))
        self.button_play.setEnabled(False)
        self.button_pause.setEnabled(False)
        self.button_play.clicked.connect(lambda: play(self))
        self.button_pause.clicked.connect(lambda: pause(self))
        self.textfield_outputText.setEnabled(False)
        self.button_Detect.clicked.connect(lambda: detect(self))
        self.button_clear.clicked.connect(lambda: clear_Text(self))
        self.lineEdit_originalFilePath.setText(
            "C:/Users/Lenovo/Desktop/Okul/CS401/Datasets/IROS2017FangOnRoadVehicle/IROS2017FangOnRoadVehicle/01.10900.csv.wmv")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
