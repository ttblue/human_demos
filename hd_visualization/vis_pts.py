from mayavi_plotter import *
import numpy as np
from mayavi import mlab


fname0 = 'library.csv'
fname  = 'points.csv'
pts0   = np.genfromtxt(fname0, delimiter=',')
pts   = np.genfromtxt(fname, delimiter=',')

plotter = PlotterInit()
req = gen_mlab_request(mlab.points3d, pts0[:,0], pts0[:,1], pts0[:,2], color=(1,0,0), scale_factor=0.05)
req1 = gen_mlab_request(mlab.points3d, pts[:,0], pts[:,1], pts[:,2], color=(0,1,0), scale_factor=0.1)

plotter.request(req)
#plotter.request(req1)

print np.linalg.norm(pts0-pts)


