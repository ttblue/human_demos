import numpy as np

def intersect_segs(ps_n2, q_22):
    """Takes a list of 2d nodes (ps_n2) of a piecewise linear curve and two points representing a single segment (q_22)
        and returns indices into ps_n2 of intersections with the segment."""
    assert ps_n2.shape[1] == 2 and q_22.shape == (2, 2)

    def cross(a_n2, b_n2):
        return a_n2[:,0]*b_n2[:,1] - a_n2[:,1]*b_n2[:,0]

    rs = ps_n2[1:,:] - ps_n2[:-1,:]
    s = q_22[1,:] - q_22[0,:]
    denom = cross(rs, s[None,:])
    qmp = q_22[0,:][None,:] - ps_n2[:-1,:]
    ts = cross(qmp, s[None,:]) / denom # zero denom will make the corresponding element of 'intersections' false
    us = cross(qmp, rs) / denom # same here
    intersections = np.flatnonzero((ts > 0) & (ts < 1) & (us > 0) & (us < 1))
    return intersections, ts, us

def rope_has_intersections(ctl_pts):
    for i in range(len(ctl_pts) - 1):
        curr_seg = ctl_pts[i:i+2,:]
        intersections, _, _ = intersect_segs(ctl_pts[:,:2], curr_seg[:,:2])
        if len(intersections) != 0:
            return True
    return False


