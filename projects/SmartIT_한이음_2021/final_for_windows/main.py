from create_UI import UI
from threading import Thread
from process import Machine
import os



def main():
    #main_UI = UI()
    #main_UI.all()
    machine = Machine()
    main_UI = UI(machine)
    a = Thread(target=machine.decision)
    a.setDaemon(True)
    a.start()
    main_UI.ui_all(machine)

if __name__ == "__main__":
    main()
