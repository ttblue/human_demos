from rapprentice import registration
from hd_utils.tps_utils import *
from hd_utils.mayavi_plotter import *

"""
Script to correct hydra-position estimates using Thin-Plate splines.
A TPS model is fit using the data when both the camera and the hydra can
see the marker.

When the camera cannot see, this TPS model is used for interpolation.
"""

def fit_tps(x_gt, x, plot=True):
    """
    Fits a thin-plate spline model to x (source points) and x_gt (ground-truth target points).
    This transform can be used to correct the state-dependent hydra errors.
    """
    bend_coef = 0.00  ## increase this to make the interpolation more smooth
    f = registration.fit_ThinPlateSpline(x, x_gt, bend_coef = bend_coef, rot_coef = 0.001)
    
    if plot:
        plot_reqs = plot_warping(f.transform_points, x, x_gt, fine=False, draw_plinks=True)
        plotter = PlotterInit()
        for req in plot_reqs:
            plotter.request(req)

    return f.transform_points
