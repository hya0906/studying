from create_UI import UI
from threading import Thread
from process import Machine
from record_data import Record_data
# PIR(0,1),마이크(0,1),온도(0,1,2),습도(0,1,2) 순

def main():
    machine = Machine()
    main_UI = UI(machine)
    a = Thread(target=machine.decision)
    a.setDaemon(True)
    a.start()
    main_UI.ui_all(machine)
    

if __name__ == "__main__":
    main()
