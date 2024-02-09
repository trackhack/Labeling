import sys
from PyQt5.QtWidgets import *
import subprocess
import os

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyQt5 GUI')
        self.setGeometry(100, 100, 300, 200)

        self.label = QLabel('Enter Cob-ID:', self)
        self.textbox = QLineEdit(self)
        self.button = QPushButton('Scan', self)

        self.button.clicked.connect(self.run_script)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.textbox)
        vbox.addWidget(self.button)

        self.setLayout(vbox)

    def run_script(self):
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            print("run_script method called")  # Check if the method is being called
            flag = self.textbox.text()

            self.textbox.clear()

            # Path to bundled executable
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.py")
            print("Path:" ,script_path)
            print("Ex:", sys.executable)
            process = subprocess.call([sys.executable, script_path, '--flag', flag])
            #stdout, stderr = process.communicate()


            if process != 0:
                print("Error: Script exited with non-zero code")
            else:
                print("Script executed successfully")
            #if stderr:
            #    print('Error:', stderr.decode())
            #else:
            #    print('Script output:', stdout.decode())
        except Exception as e:
            print("Exception occurred in run_script:", e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
