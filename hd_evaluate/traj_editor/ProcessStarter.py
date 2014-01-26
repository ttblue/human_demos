from PyQt4     import   QtCore, QtGui
from pnpApp   import   pnpApp
from multiprocessing import Process,Pipe
from ORServer  import   *
import sys


class ProcessStarter(object):
    
    def __init__(self):
        '''
        Setup the shared memory data structure model and initialize the control parts.
        '''
        self.running = True
        self.orprocess  = None
        self.guiprocess = None
        self.pipeGUI, self.pipeOR = Pipe()
        self.StartProcesses()

    def StartProcesses(self):
        self.guiprocess = Process(target=self.__startGUI__)
        self.guiprocess.start()        
        self.pipeGUI.send(["StartViewer", None])
        self.orprocess = Process(target=ORServer,args=(self.pipeOR,))
        self.orprocess.start()
        return True
    
    def terminate(self):
        try:
            self.pipeGUI.send(["Stop", None])
            self.guiprocess.terminate()
            self.orprocess.terminate()
        except:
            pass

    def __startGUI__(self):
        app  = QtGui.QApplication(sys.argv)
        form = pnpApp(self.pipeGUI, self)
        form.show()
        sys.exit(app.exec_())
