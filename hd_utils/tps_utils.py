import numpy as np
from hd_utils.colorize import colorize
from mayavi import mlab
from hd_utils.mayavi_plotter import *

"""
Utility functions to visualize tps warping function.
"""

def gen_grid(f, mins, maxes, ncoarse=10, nfine=30):
    """
    generate 3d grid and warps it using the function f.
    The grid is based on the number of lines (ncoarse & nfine).
    """    
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    xcoarse = np.linspace(xmin, xmax, ncoarse)
    ycoarse = np.linspace(ymin, ymax, ncoarse)
    zcoarse = np.linspace(zmin, zmax, ncoarse)

    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    zfine = np.linspace(zmin, zmax, nfine)
    
    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((nfine, 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    return lines


def gen_grid2(f, mins, maxes, xres = .01, yres = .01, zres = .01):
    """
    generate 3d grid and warps it using the function f.
    The grid is based on the resolution specified.
    """    
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    xcoarse = np.arange(xmin, xmax+xres/10., xres)
    ycoarse = np.arange(ymin, ymax+yres/10., yres)
    zcoarse = np.arange(zmin, zmax+zres/10., zres)
    
    xfine = np.arange(xmin, xmax+xres/10., xres/5.)
    yfine = np.arange(ymin, ymax+yres/10., yres/5.)
    zfine = np.arange(zmin, zmax+zres/10., zres/5.)

    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((len(zfine), 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((len(xfine), 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((len(yfine), 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    return lines


def plot_warping(f, src, target, fine=True, draw_plinks=True):
    """
    function to plot the warping as defined by the function f.
    src    : nx3 array
    target : nx3 array
    fine   : if fine grid else coarse grid.
    """
    print colorize("Plotting grid ...", 'blue', True)
    mean  = np.mean(src, axis=0)
    mins  = np.min(src, axis=0)
    maxes = np.max(src, axis=0)

    grid_lines = []
    if fine:
        grid_lines = gen_grid2(f, mins=mins, maxes=maxes, xres=0.005, yres=0.005, zres=0.002)
    else:
        grid_lines = gen_grid(f, mins=mins, maxes=maxes)

    
    plotter_requests = []
    plotter_requests.append(gen_mlab_request(mlab.clf))
    plotter_requests.append(gen_custom_request('lines', lines=grid_lines, color=(0,0.5,0.3)))
    
    warped = f(src)
    
    plotter_requests.append(gen_mlab_request(mlab.points3d, src[:,0], src[:,1], src[:,2], color=(1,0,0), scale_factor=0.001))
    plotter_requests.append(gen_mlab_request(mlab.points3d, target[:,0], target[:,1], target[:,2], color=(0,0,1), scale_factor=0.001))
    plotter_requests.append(gen_mlab_request(mlab.points3d, warped[:,0], warped[:,1], warped[:,2], color=(0,1,0), scale_factor=0.001))

    if draw_plinks:
        plinks = [np.c_[ps, pw].T for ps,pw in zip(src, warped)]
        plotter_requests.append(gen_custom_request('lines', lines=plinks, color=(0.5,0,0), line_width=2, opacity=1))
                                
    return plotter_requests
