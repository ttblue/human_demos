import serial 
import glob
from hd_utils.colorize import colorize
import time
from threading import Thread

from hd_utils.func_utils import once

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
            for n,d in enumerate(devices):
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
        """
        buffer = ''
        while True:
            buffer = buffer + self.ser.read(self.ser.inWaiting())
            if '\n' in buffer:
                lines = buffer.split('\n')
                self.reading = lines[-2]
                buffer = lines[-1]


    def get_reading(self, idx=1):
        """
        return the latest reading read from the arduino.
        """
        while True:
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
