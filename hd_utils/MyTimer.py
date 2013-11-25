from threading import Thread
import time

class MyTimer(Thread):
    def __init__(self, t, func):
        self.stopped = False
        Thread.__init__(self)
        self.func = func
        self.t    = t
        self.start()

    def run(self):
        while not self.stopped:
            self.func()
            time.sleep(self.t)