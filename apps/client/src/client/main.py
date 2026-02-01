import sys
import os

# No need for manual sys.path manipulation if running through uv or correctly installed
from PyQt6.QtWidgets import QApplication
from client.ui.main_window import MainWindow
from dotenv import load_dotenv

def main():
    load_dotenv()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
