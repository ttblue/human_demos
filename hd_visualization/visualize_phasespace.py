import time
from mayavi import mlab
import numpy as np
import threading as th


class threadClass (th.Thread):
    
    def __init__ (self):
        th.Thread.__init__(self)
        self.handle = None
     
    def update_handle (self, handle):
        self.handle = handle
        
    def update_x (self, x):
        handle.mlab_source.x = x
    
    def update_y (self, y):
        handle.mlab_source.y = y
        
    def update_z (self, z):
        handle.mlab_source.z = z
        
    def run (self):
        print 1
        mlab.show()


def playback_log (log_file):
    import yaml, os.path as osp
    
    log_loc = "/home/sibi/sandbox/human_demos/hd_data/phasespace_logs" 
    with open(osp.join(log_loc,logfile)) as fh: marker_pos = yaml.Load(fh)
    
    
def test_mayavi ():

    thc = threadClass ()
    x = np.reshape(range(12),[4,3])
    s = [1,1,1,1]
    h = mlab.points3d(x[:,0], x[:,1], x[:,2], color=(1,0,0))
    
    thc.start()
    
    ms = h.mlab_source
    i = 0
    t = np.linspace(0, 4*np.pi, 4)
    while True:
        try:
            #if i:
            ms.z = - ms.z#np.array([0,1,2,3])
            #i = 1
            #else:
            #    ms.z = np.array([0,-1,-2,-3])
            #    i = 0
#             ms.z = np.cos(2*t*0.1*(i+1))
#             i = np.mod(1+i,10)
#             time.sleep(0.25)
        except KeyboardInterrupt:
            print "done"
            break
        
        
def test_points3d():
    t = np.linspace(0, 4*np.pi, 20)
    cos = np.cos
    sin = np.sin

    x = sin(2*t)
    y = cos(t)
    z = cos(2*t)
    s = 2+sin(t)

    return mlab.points3d(x, y, z, s, colormap="copper", scale_factor=.25)

def test_points3d_anim():
    """Animates the test_points3d example."""
    g = test_points3d()
    t = np.linspace(0, 4*np.pi, 20)
    # Animate the points3d.
    ms = g.mlab_source
    for i in range(10):
        ms.z = np.cos(2*t*0.1*(i+1))
    return g
