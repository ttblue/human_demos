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


class pnpApp(QtGui.QMainWindow, pnp_ui.Ui_MainWindow):

    @QtCore.pyqtSignature("")
    def on_pbClose_clicked(self):
        self.closeAll()


    def __init__(self, env_state_dir):
        super(pnpApp, self).__init__(None)
        self.setupUi(self)
                
        self.env_state_dir = env_state_dir
        self.env_state_files = glob.glob(osp.join(env_state_dir, "*.cp"))
        self.num_tests       = len(self.env_state_files)
        self.results_file    = osp.join(env_state_dir, 'results.cpkl')
        
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

        if self.runnum+1 == self.num_tests:
            self.done = True
            self.runLabel.setText('%d/%d'%(self.runnum+1, self.num_tests))
            cPickle.dump(self.results, open(self.results_file,'w'))
        else:
            self.runnum +=1
            self.runLabel.setText('%d/%d'%(self.runnum, self.num_tests))
            self.load_image()

    def load_image(self):
        img_fname = osp.join(self.env_state_dir, self.get_demo_name(), 'composite.jpg')
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
    parser.add_argument("--demo_type", type=str)
    args = parser.parse_args()
    
    env_state_dir   = osp.join(demo_files_dir, args.demo_type, 'test_env_states')
    
    app  = QtGui.QApplication(sys.argv)
    form = pnpApp(env_state_dir)
    form.show()
    sys.exit(app.exec_())
