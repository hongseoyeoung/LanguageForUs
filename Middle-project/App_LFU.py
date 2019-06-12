import sys
import subprocess

from PyQt5.QtWidgets import *



class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LFU")
        self.setGeometry(300, 300, 300, 400)

        btn1 = QPushButton("손 인식", self)
        btn1.setStyleSheet("border: 1px solid black; background: yellow")
        btn1.move(40, 40)
        btn1.clicked.connect(self.btn1_clicked)

        btn2 = QPushButton("화면 출력", self)
        btn2.setStyleSheet("border: 1px solid black; background: skyblue")
        btn2.move(160, 40)
        btn2.clicked.connect(self.btn2_clicked)

    #손인식 버튼 클릭시
    def btn1_clicked(self):

        #window
        tmp = subprocess.Popen("set_hand_hist.py", shell=True)
        tmp.communicate()

        #mac
       # subprocess.call(['python set_hand_hist.py'], shell=True)

    #화면 출력 버튼 클릭시
    def btn2_clicked(self):

        # window
        tmp = subprocess.Popen("python recognize_gesture.py", shell=True)
        tmp.communicate()

        #mac
       # subprocess.call(['python recognize_gesture.py'], shell=True)

    #close 이벤트
    #yes일때 창 종료 No일때 창 종료 무시
    def closeEvent(self, QCloseEvent):
        ans = QMessageBox.question(self, "종료확인", "종료하시겠습니까?",
                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if ans == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()