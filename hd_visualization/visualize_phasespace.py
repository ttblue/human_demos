import time
from mayavi import mlab
import numpy as np

import mayavi_plotter as mp

def playback_log (log_file):
    import yaml, os.path as osp
    
    log_loc = "/home/sibi/sandbox/human_demos/hd_data/phasespace_logs" 
    with open(osp.join(log_loc,log_file),"r") as fh: marker_pos = yaml.load(fh)
    
    #handle = mlab.points3d([0],[0],[0], color = (1,0,0), scale_factor = 0.25)
    #ms = handle.mlab_source
    plotter = mp.PlotterInit()
    
    prev_time = time.time()
    for step in marker_pos:
        
        markers = np.asarray(step['marker_positions'].values())
        
        clear_req = mp.gen_mlab_request(mlab.clf)
        plot_req = mp.gen_mlab_request(mlab.points3d, markers[:,0], markers[:,1], markers[:,2], color=(1,0,0),scale_factor=0.25)
            
        time_to_wait = max(step['time_stamp'] - time.time() + prev_time,0.0)
        time.sleep(time_to_wait)
        
        plotter.request(clear_req)
        plotter.request(plot_req)
        
        prev_time = time.time()
        
    print "Playback of log file %s finished."%log_file
