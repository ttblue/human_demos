import serial 

ser = serial.Serial('/dev/ttyACM1', 9600)
while True:
    print ser.readline()

