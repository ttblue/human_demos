#! /usr/bin/env python

from PyQt4 import QtCore, QtGui
import pnp_ui 
import argparse
import yaml, sys
from hd_utils.colorize import colorize
import cPickle

import os, numpy as np, h5py, time, os.path as osp
import cPickle as cp
from numpy.linalg import norm
import glob

from hd_utils import yes_or_no
from hd_utils.utils import make_perp_basis
from hd_utils.colorize import *
from hd_utils.defaults import demo_files_dir, hd_data_dir, cad_files_dir
import time
import  hd_rapprentice.cv_plot_utils as cpu

from hd_utils.defaults import testing_results_dir



class pnpApp(QtGui.QMainWindow, pnp_ui.Ui_MainWindow):

    @QtCore.pyqtSignature("")
    def on_pbClose_clicked(self):
        self.closeAll()


    def __init__(self, snapshot_path, save_path):
        super(pnpApp, self).__init__(None)
        self.setupUi(self)
                
                
        self.env_state_dir =  snapshot_path
        
        self.env_state_files = glob.glob(osp.join(self.env_state_dir,"*.jpg"))
        self.num_tests       = len(self.env_state_files)
        self.results_file    = save_path
        
        self.done = False
        self.runnum = 0
        self.results = {} 
        
        self.add_slots()
        
        # init display
        self.load_image()
        self.runLabel.setText('%d/%d'%(self.runnum+1, self.num_tests))


    def add_slots(self):
        QtCore.QObject.connect(self.failButton, QtCore.SIGNAL("clicked()"), self.clicked_failButton)
        QtCore.QObject.connect(self.pnpButton,  QtCore.SIGNAL("clicked()"), self.clicked_pnpButton)
        QtCore.QObject.connect(self.passButton, QtCore.SIGNAL("clicked()"), self.clicked_passButton)

    def get_demo_name(self):
        return osp.basename(osp.splitext(self.env_state_files[self.runnum])[0])

    def update_runnums(self, res):
        self.results[self.get_demo_name()] = res

        if self.runnum+1 == self.num_tests or self.runnum >=2:
            self.done = True
            self.runLabel.setText('%d/%d'%(self.runnum+1, self.num_tests))
            cPickle.dump(self.results, open(self.results_file,'w'))
            sys.exit(0)
        else:
            self.runnum +=1
            self.runLabel.setText('%d/%d'%(self.runnum, self.num_tests))
            self.load_image()

    def load_image(self):
        img_fname = self.env_state_files[self.runnum]
        pixmap    = QtGui.QPixmap(img_fname)
        if pixmap.isNull():
            print "File not found : %s"%img_fname
        self.imgLabel.setPixmap(pixmap)


    def clicked_failButton(self):
        if not self.done:
            self.update_runnums(0)
        
    def clicked_pnpButton(self):
        if not self.done:
            self.update_runnums(0.5)
        
    def clicked_passButton(self):
        if not self.done:
            self.update_runnums(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="P/NP App")
    parser.add_argument("--snapshot_path", type=str)
    parser.add_argument("--save_path", type=str)
    
    args = parser.parse_args()

    
    app  = QtGui.QApplication(sys.argv)
    form = pnpApp(args.snapshot_path, args.save_path)
    form.show()
    sys.exit(app.exec_())
