from PyQt4          import QtCore, QtGui
import sys, os, re, logging, signal, time, os.path as osp
from numpy import random
from multiprocessing import Process,Pipe
from threading import Thread
import pnp_ui
import numpy as np
from RopePR2Viz        import RopePR2Viz
from ProcessStarter import *
from joints_utils import *
import copy
from hd_utils.colorize import colorize

from sceneparser import *
import cPickle    

class pnpApp(QtGui.QMainWindow, pnp_ui.Ui_MainWindow):

    @QtCore.pyqtSignature("")
    def on_pbClose_clicked(self):
        self.closeAll()

    def __init__(self, pipeOR, processStarter):
        super(pnpApp,self).__init__(None)
        self.pipeOR = pipeOR        
        self.setupUi(self)
        self.processStarter = processStarter
        self.results = {}
        self.addSlots()
        self.done = False

    def get_random_rope_nodes(self, n=10):
        """
        return nx3 matrix of rope nodes:
        """
        nodes = np.c_[np.linspace(0,1,n), np.linspace(0,1,n), np.linspace(0,1,n)]
        nodes += np.random.randn(n,3)/20.
        return nodes

    def get_random_dofs_tfm(self):
        return np.random.randn(39)/50, np.eye(4)


    def closeEvent(self, event):
        """
        Redefine the close event.
        """
        QtGui.QApplication.quit()
        self.processStarter.terminate()

        
    def addSlots(self):
        QtCore.QObject.connect(self.failButton, QtCore.SIGNAL("clicked()"), self.clicked_failButton)
        QtCore.QObject.connect(self.pnpButton,  QtCore.SIGNAL("clicked()"), self.clicked_pnpButton)
        QtCore.QObject.connect(self.passButton, QtCore.SIGNAL("clicked()"), self.clicked_passButton)


    def update_runnums(self, res):
        #self.results[int(self.runnum)] = res
        self.updateRave()
        
        """
        if self.runnum == self.run_range[self.rangenum, 1]:
            self.save_results()
            self.results = {}
            self.rangenum += 1
            if self.rangenum >= self.run_range.shape[0]:
                self.done = True
            else:
                self.runnum = self.run_range[self.rangenum, 0]
        else:
            self.runnum += 1
            
        if not self.done: # update the picture
            self.runLabel.setText('%d/%d'%(int(self.runnum), int(self.run_range[self.rangenum,1])))
            self.load_image()
        """
           

    def save_results(self):       
        pass

    def clicked_failButton(self):
        print "fail"
        if not self.done:
            self.update_runnums(0)

    def clicked_pnpButton(self):
        print "not sure"
        if not self.done:
            self.update_runnums(0.5)

    def clicked_passButton(self):
        print "pass!!"
        if not self.done:
            self.update_runnums(1)

    def updateRave(self):
        self.pipeOR.send(['SetRobotPose', cPickle.dumps(self.get_random_dofs_tfm())])
        self.pipeOR.send(['UpdateRope', cPickle.dumps(self.get_random_rope_nodes())])


if __name__ == "__main__":
    try:
        ProcessStarter()
    except:
        sys.exit(0)

