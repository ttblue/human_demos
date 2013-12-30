from hd_utils.colorize import colorize
from hd_utils.func_utils import once
from threading import Thread
import glob
import serial
import time


class Arduino:
    """
    Simple class to read the integer put on the serial port by the arduino.
    """

    def __init__(self):
        devices = glob.glob("/dev/ttyACM*")
        if len(devices) ==0:
            raise RuntimeError(colorize( "Arduino not connected.", 'red', True))
        elif len(devices) >1:
            print colorize("Found more than one matching devices. Enter the number of the one which is arduino:", 'blue', True)
            for n, d in enumerate(devices):
                print "\t%d. %s"%(n,d)
                while True:
                    try:
                        sel = int(raw_input())
                        if 0<= sel < len(devices):
                            break  
                    except:
                        print "\tPlease enter a number in the range [0, %d]" % len(devices)
                        pass
                arduino = devices[sel]
        else:           
            arduino = devices[0]
        
        try:
            self.ser = serial.Serial(arduino, 9600, timeout=1)
        except serial.SerialException as e:
            raise RuntimeError(colorize("Error opening serial port '{}': {}".format(arduino, e), "red", True))

        self.poll_thread = Thread(target=self.poll_arduino)
        self.poll_thread.start()
        
        time.sleep(0.1)
        print colorize("Arduino all set.", "green", True)


    def poll_arduino(self):
        """
        Polls the arduino in a separate thread.
        Gives old reading until valid new reading is found.
        """
        buffer = ''
        while True:
            new_reading = self.ser.readline()
            
            try:
                new_vals = [int(val) for val in new_reading.split()]
                if new_vals[0]^new_vals[1] == new_vals[2]:
                    self.reading = new_reading
                else: 
                    print "Got bad reading [v1, v2, v1 XOR v2]: ", new_reading
            except:
                pass
                    


    def get_reading(self, idx=1):
        """
        return the latest reading read from the arduino.
        """
        try:
            return int(self.reading.split()[idx-1])
        except:
            pass

@once
def get_arduino ():
    ard = Arduino()
    return ard

if __name__ == '__main__':
    ard = Arduino()
    while True:
        print "reading : ", ard.get_reading()
        time.sleep(.01)
