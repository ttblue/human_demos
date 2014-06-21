"""
Register point clouds to each other


arrays are named like name_abc
abc are subscripts and indicate the what that tensor index refers to

index name conventions:
    m: test point index
    n: training point index
    a: input coordinate
    g: output coordinate
    d: gripper coordinate
"""

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd

from hd_utils import math_utils

from hd_rapprentice import tps, svds 



class Transformation(object):
    """
    Object oriented interface for transformations R^d -> R^d
    """
    def transform_points(self, x_ma):
        raise NotImplementedError
    def compute_jacobian(self, x_ma):
        raise NotImplementedError        

        
    def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """

        grad_mga = self.compute_jacobian(x_ma)
        newrot_mgd = np.array([grad_ga.dot(rot_ad) for (grad_ga, rot_ad) in zip(grad_mga, rot_mad)])
        

        if orthogonalize:
            if orth_method == "qr": 
                newrot_mgd =  orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd
        
    def transform_hmats(self, hmat_mAD):
        """
        Transform (D+1) x (D+1) homogenius matrices
        """
        hmat_mGD = np.empty_like(hmat_mAD)
        hmat_mGD[:,:3,3] = self.transform_points(hmat_mAD[:,:3,3])
        hmat_mGD[:,:3,:3] = self.transform_bases(hmat_mAD[:,:3,3], hmat_mAD[:,:3,:3])
        hmat_mGD[:,3,:] = np.array([0,0,0,1])
        return hmat_mGD
        
    def compute_numerical_jacobian(self, x_d, epsilon=0.0001):
        "numerical jacobian"
        x0 = np.asfarray(x_d)
        f0 = self.transform_points(x0)
        jac = np.zeros(len(x0), len(f0))
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (self.transform_points(x0+dx) - f0) / epsilon
            dx[i] = 0.
        return jac.transpose()

class ThinPlateSpline(Transformation):
    """
    members:
        x_na: centers of basis functions
        w_ng: 
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    
    """
    def __init__(self, d=3):
        "initialize as identity"
        self.x_na = np.zeros((0,d))
        self.lin_ag = np.eye(d)
        self.trans_g = np.zeros(d)
        self.w_ng = np.zeros((0,d))
        self.z_scale = 1
        self.corr_nm = None
        self.is_2d = False

    def transform_points(self, x_ma):
        #import IPython; IPython.embed()
        #x_ma[:,-1] = x_ma[:,-1]*self.z_scale
        if not self.is_2d:
            _,d = x_ma.shape
            y_ng = tps.tps_eval(x_ma, self.lin_ag[:d,:d], self.trans_g[:d], self.w_ng[:,:d], self.x_na[:,:d])
            y_ng[:,-1] = y_ng[:,-1]/self.z_scale
        else:
            print "transforming 3d points"
            #import IPython; IPython.embed()
            _,d = x_ma.shape
            y_ng = tps.tps_eval_2d(x_ma, self.lin_ag[:d,:d], self.trans_g[:d], self.w_ng[:,:d], self.x_na[:,:d])
            y_ng[:,-1] = y_ng[:,-1]/self.z_scale
        return y_ng

    def compute_jacobian(self, x_ma):
        grad_mga = tps.tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga
        
class Affine(Transformation):
    def __init__(self, lin_ag, trans_g):
        self.lin_ag = lin_ag
        self.trans_g = trans_g
    def transform_points(self, x_ma):
        return x_ma.dot(self.lin_ag) + self.trans_g[None,:]  
    def compute_jacobian(self, x_ma):
        return np.repeat(self.lin_ag.T[None,:,:],len(x_ma), axis=0)
        
class Composition(Transformation):
    def __init__(self, fs):
        "applied from first to last (left to right)"
        self.fs = fs
    def transform_points(self, x_ma):
        for f in self.fs: x_ma = f.transform_points(x_ma)
        return x_ma
    def compute_jacobian(self, x_ma):
        grads = []
        for f in self.fs:
            grad_mga = f.compute_jacobian(x_ma)
            grads.append(grad_mga)
            x_ma = f.transform_points(x_ma)
        totalgrad = grads[0]
        for grad in grads[1:]:
            totalgrad = (grad[:,:,:,None] * totalgrad[:,None,:,:]).sum(axis=-2)
        return totalgrad

def fit_ThinPlateSpline(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None, K_nn=None):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points
    x_ta: penalize distance of these points in the z-axis
    """
    #import tn_rapprentice.tps as tn_tps
    x_na = np.array(x_na, dtype='float64')
    y_ng = np.array(y_ng, dtype='float64')
    if wt_n != None:
        wt_n = np.array(wt_n, dtype='float64')

    bend_coef = np.array([bend_coef*10e0, bend_coef*10e0, bend_coef*10e-1])

    f = ThinPlateSpline()
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    #f.lin_ag, f.trans_g, f.w_ng = tn_tps.tps_fit3_cvx(x_na, y_ng, bend_coef, rot_coef, wt_n)
    f.x_na = x_na
    return f        
   
def fit_ThinPlateSpline2d(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None, K_nn=None):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points        
    """
    f = ThinPlateSpline()
    f.lin_ag[:-1,:-1], f.trans_g[:-1], new_w_ng = tps.tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    f.w_ng = np.hstack([new_w_ng, np.zeros((len(new_w_ng),1))])
    f.x_na = np.hstack([x_na, np.zeros((len(x_na),1))])
    return f   

def fit_ThinPlateSpline_RotReg(x_na, y_ng, bend_coef = .1, rot_coefs = (0.01,0.01,0.0025),scale_coef=.01):
    import fastrapp
    f = ThinPlateSpline()
    rfunc = fastrapp.rot_reg
    fastrapp.set_coeffs(rot_coefs, scale_coef)
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit_regrot(x_na, y_ng, bend_coef, rfunc)
    f.x_na = x_na
    return f        


def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    


def unit_boxify(x_na):    
    ranges = x_na.ptp(axis=0)
    dlarge = ranges.argmax()
    unscaled_translation = - (x_na.min(axis=0) + x_na.max(axis=0))/2
    scaling = 1./ranges[dlarge]
    scaled_translation = unscaled_translation * scaling
    return x_na*scaling + scaled_translation, (scaling, scaled_translation)
    
def unscale_tps_3d(f, src_params, targ_params):
    """Only works in 3d!!"""
    try:
        assert len(f.trans_g) == 3
        p,q = src_params
        r,s = targ_params
        #print p,q,r,s
        fnew = ThinPlateSpline()
        fnew.x_na = (f.x_na  - q[None,:])/p 
        fnew.w_ng = f.w_ng * p / r
        fnew.lin_ag = f.lin_ag * p / r
        fnew.trans_g = (f.trans_g  + f.lin_ag.T.dot(q) - s)/r

        fnew.is_2d = f.is_2d
        fnew.corr_nm = f.corr_nm #tkl remove
    except Exception as exc:
        print exc; import IPython; IPython.embed()    
    return fnew

def unscale_tps(f, src_params, targ_params):
    """Only works in 3d!!"""
    p,q = src_params
    r,s = targ_params
    
    d = len(q)
    
    lin_in = np.eye(d)*p
    trans_in = q
    aff_in = Affine(lin_in, trans_in)
    
    lin_out = np.eye(d)/r
    trans_out = -s/r
    aff_out = Affine(lin_out, trans_out)

    return Composition([aff_in, f, aff_out])
    
    

def tps_rpm(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg=1e-4,
            plotting = False, f_init = None, plot_cb = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)

        wt_n = corr_nm.sum(axis=1)


        targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_nd, corr_nm, wt_n, f)
        
        
        f = fit_ThinPlateSpline(x_nd, targ_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)

    return f


def tps_rpm_bij_switch(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
                plotting = False, plot_cb = None, critical_points=0 , added_pts=0):
    #if added_pts != 0:
    return tps_rpm_bij_3d(x_nd, y_md, reg_init=reg_init, reg_final=reg_final, rad_init=rad_init, rad_final=rad_final,
            rot_reg=rot_reg, n_iter=n_iter, critical_points=critical_points, added_pts=added_pts)
    # else:
    #     return tps_rpm_bij_2d(x_nd, y_md, reg_init=reg_init, reg_final=reg_final, rad_init=rad_init, rad_final=rad_final,
    #         rot_reg=rot_reg, n_iter=n_iter, critical_points=critical_points)


def tps_rpm_bij(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
                plotting = False, plot_cb = None, block_lengths=None, Globals=None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    import traceback

    if block_lengths == None: block_lengths = [(len(x_nd), len(y_md))]

    #adjust for duplicate points to prevent singular matrix errors
    for pass_num in range(1):
        for i in range(len(x_nd)):
            for j in range(len(x_nd)):
                if i!=j and np.all(x_nd[i]==x_nd[j]):
                    x_nd[j][0] += 0.000001

    _,d=x_nd.shape

    # if n_iter%30 == 0:
    #     regs = loglinspace(reg_init, reg_final, 30)
    #     rads = loglinspace(rad_init, rad_final, 30)
    #     regs = np.vstack([regs[:,None] for i in range(int(n_iter/30))])
    #     regs = np.vstack([rads[:,None] for i in range(int(n_iter/30))])
    # elif n_iter > 40:
    #     regs = loglinspace(reg_init, reg_final, 30)
    #     rads = loglinspace(rad_init, rad_final, 30)
    #     regs = np.vstack([regs[:,None], np.repeat(regs[-1], n_iter-30)[:,None]])
    #     regs = np.vstack([rads[:,None], np.repeat(rads[-1], n_iter-30)[:,None]])
    # else:
    #     regs = loglinspace(reg_init, reg_final, n_iter)
    #     rads = loglinspace(rad_init, rad_final, n_iter)

    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g

    # r_N = None

    x_nd_full = np.array(x_nd)
    y_md_full = np.array(y_md)
    x_nd = np.array(x_nd[:block_lengths[0][0]])
    y_md = np.array(y_md[:block_lengths[0][0]])
    handles = []
    from scipy.spatial import Voronoi
    vorx = Voronoi(x_nd[:block_lengths[0][0]][:,:2])


    for i in xrange(n_iter):
        if i > -1:
           x_nd = x_nd_full
           y_md = y_md_full
        
        try:
            r = rads[i]
        except Exception as err:
            import IPython; IPython.embed()

        prob_nm = block_prob_nm(f, g, x_nd, y_md, block_lengths, r, i)

        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 2e-1)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)
        
        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)

        if i > -1 and len(block_lengths) > 1: #set weights for table point matching
           wt_n, wt_m = adjust_weights(wt_n, wt_m, block_lengths)

        try:
            f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
            g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)
        except:
            tb = traceback.format_exc()
            print "error in tps_rpm_bij"
            import IPython; IPython.embed() #import pdb; pdb.set_trace()

        # if Globals and len(block_lengths) > 1:
        #     print "interation", i, "of", n_iter
        #     plot_warp_progress(x_nd, y_md,f, block_lengths, Globals)
        vort = Voronoi(f.transform_points(x_nd[:block_lengths[0][0]])[:,:2])

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[-1], wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[-1], wt_n=wt_m)/wt_m.mean()

    f.corr_nm = corr_nm

    # if len(block_lengths) > 1 and Globals:
    #     import IPython; IPython.embed()

    return f,g


def block_prob_nm(f, g, x_nd, y_md, block_lengths, r,i):
    try:
        prob_nm = np.zeros((len(x_nd),len(y_md)))
        oldb = (0,0)
        for xb,yb in block_lengths:
            x_bd = x_nd[oldb[0]:oldb[0]+xb]
            y_bd = y_md[oldb[1]:oldb[1]+yb]
            xwarped_bd = f.transform_points(x_bd)
            ywarped_bd = g.transform_points(y_bd)
            fwddist_bb = ssd.cdist(xwarped_bd, y_bd,'euclidean')
            invdist_bb = ssd.cdist(x_bd, ywarped_bd,'euclidean')
            prob_nm[oldb[0]:oldb[0]+xb,oldb[1]:oldb[1]+yb] = np.exp( -(fwddist_bb + invdist_bb) / (2*r) )
            oldb = oldb[0]+xb, oldb[1]+yb
    except Exception as exc:
        print "error in block_prob_nm"
        print exc; import IPython; IPython.embed()

    try:
        for j in range(block_lengths[0][0]):
            prob_nm[j,:] = 0
            prob_nm[:,j] = 0
            prob_nm[j,j] = 1
        # if i > 4 and len(block_lengths) > 1:
        #     for j in range(block_lengths[0][0]+block_lengths[1][0]):
        #         prob_nm[j,:] = 0
        #         prob_nm[:,j] = 0
        #         prob_nm[j,j] = 1
    except Exception as exc:
        print "error in identifying matrix"
        import IPython; IPython.embed()
    # if i == 0 and len(x_nd) == len(y_md): #initialize correspondence
    #    for j in range(len(x_nd)):
    #        prob_nm[j,:] = 0
    #        prob_nm[:,j] = 0
    #        prob_nm[j,j] = 1

    return prob_nm


def adjust_weights(wt_n, wt_m, block_lengths):
    wt_n = np.repeat(wt_n[:, None], 3, axis=1)
    wt_n[-block_lengths[-1][0]-1:,:2] *= 0.0 #ignore x and y distance for table points
    wt_n[-block_lengths[-1][0]-1:,2] *= 10000

    wt_m = np.repeat(wt_m[:, None], 3, axis=1)
    wt_m[-block_lengths[-1][1]-1:,:2] *= 0.0 #ignore x and y distance for table points
    wt_m[-block_lengths[-1][1]-1:,2] *= 10000

    return wt_n, wt_m


def plot_warp_progress(x_nd, y_md,f, block_lengths, Globals, handles=None):
    if handles==None:
        handles=[]
    old_xyz = x_nd[:block_lengths[0][0]]; new_xyz = y_md[:block_lengths[0][1]]
    range1 = range(block_lengths[0][1])
    range2 = range(block_lengths[0][0]+block_lengths[1][0],2*block_lengths[0][0]+block_lengths[1][0])
    range3 = range(2*block_lengths[0][0]+block_lengths[1][0],block_lengths[0][0]+block_lengths[1][0]+block_lengths[2][0])
    for rangei in [range1]: #, range2, range3]:
        handles.append(Globals.env.plot3(x_nd[rangei], 7, np.array([(1,0,0,1) for i in x_nd[rangei]])))
        handles.append(Globals.env.plot3(y_md[rangei], 7, np.array([(0,0,1,1) for i in y_md[rangei]])))
        handles.append(Globals.env.plot3(f.transform_points(x_nd[rangei]), 7, np.array([(0,1,0,1) for i in x_nd[rangei]])))
        handles.append(Globals.env.drawlinestrip(x_nd[rangei],5,(5,0,1,1)))
        handles.append(Globals.env.drawlinestrip(y_md[rangei],5,(0,1,5,1)))
        handles.append(Globals.env.drawlinestrip(f.transform_points(x_nd[rangei]),5,(1,5,0,1)))

    pair_list = [(-1,31),(-7,30),(-8,29),(-9,28)]#[(8,60),(7,61),(9,59),(0,66)]#[(6,25),(2,9),(32,36)]#[(6,27),(6,25),(7,23),(5,27),(5,28),(13,31),(15,30)]
    color_list = [(1,0,1,1),(1,0,1,1),(1,0,1,1),(1,0,1,1)]#,(1,0,1,1),(1,2,1,1),(1,2,1,1)]
    for pair, color in zip(pair_list, color_list):
        orig_diff = np.array([(i/40.)*(old_xyz[pair[0]])+(1-i/40.)*old_xyz[pair[1]] for i in range(41)])  #8,58 for demo33; [(6,25),(2,9),(32,36)] for demo09 
        handles.append(Globals.env.plot3(orig_diff, 10, np.array([(1,0,1,1) for i in orig_diff])))
        tfmd_diff = f.transform_points(orig_diff)
        handles.append(Globals.env.plot3(tfmd_diff, 10, np.array([color for i in orig_diff])))
        handles.append(Globals.env.drawlinestrip(tfmd_diff,5,(3,1,0,1)))
    # orig_diff = np.array([(i/40.)*(old_xyz[-4])+(1-i/40.)*.5*(old_xyz[-1]+old_xyz[31]) for i in range(41)])  #8,58 for demo33; [(6,25),(2,9),(32,36)] for demo09 
    # handles.append(Globals.env.plot3(orig_diff, 10, np.array([(1,0,1,1) for i in orig_diff])))
    # tfmd_diff = f.transform_points(orig_diff)
    # handles.append(Globals.env.plot3(tfmd_diff, 10, np.array([color for i in orig_diff])))
    # handles.append(Globals.env.drawlinestrip(tfmd_diff,5,(3,1,0,1)))

    Globals.viewer.Idle()
    #import IPython; IPython.embed()

def plot_local_area(x_nd, f, ind, handles, Globals, clear_prev=0):
    c1,c2 = get_local_area(x_nd, f, ind)
    for i in handles[-clear_prev-1:][:-1]: 
        i.Close()
    handles.append(Globals.env.plot3(c1, 10, np.array([(1,0,0,1),(0,1,0,1),(1,0,0,1),(0,0,1,1)])))
    handles.append(Globals.env.plot3(c2, 10, np.array([(1,0,0,1),(0,1,0,1),(1,0,0,1),(0,0,1,1)])))
    handles.append(Globals.env.drawlinestrip(np.vstack([c1,c1[0]]),5,(1,1,0,1)))
    handles.append(Globals.env.drawlinestrip(np.vstack([c2,c2[0]]),5,(1,1,0,1)))


def plot_voronoi_region(vor, f, ind, handles, Globals, clear_prev=0):
    c1 = vor.regions[vor.point_region[ind]]
    handles.append(Globals.env.plot3(np.hstack([vor.vertices[c1], np.zeros((len(c1),1))]), 10, np.array([(1,0,0,1) for i in c1])))
    handles.append(Globals.env.plot3(f.transform_points(np.hstack([vor.vertices[c1], np.zeros((len(c1),1))])), 10, np.array([(1,0,0,1) for i in c1])))
    c1.append(c1[0])
    handles.append(Globals.env.drawlinestrip(np.hstack([vor.vertices[c1], np.zeros((len(c1),1))]),5,(1,1,0,1)))
    handles.append(Globals.env.drawlinestrip(f.transform_points(np.hstack([vor.vertices[c1], np.zeros((len(c1),1))])),5,(1,1,0,1)))
    Globals.viewer.Step()


def get_local_area(x_nd, f, ind):
    c1 = np.array([x_nd[ind] + i for i in [[0,.05,0],[.05,0,0],[0,-.05,0],[-.05,0,0]]])
    #(x1y2 - x2y1) + (x2y3 - x3y2) + (x3y4 - x4y3) + (x4y1 - x1y4)
    area1 = 0.5 *abs((c1[0,0]*c1[1,1] - c1[1,0]*c1[0,1]) + (c1[1,0]*c1[2,1] - c1[2,0]*c1[1,1])+ (c1[2,0]*c1[3,1] - c1[3,0]*c1[2,1])+ (c1[3,0]*c1[0,1] - c1[0,0]*c1[3,1]))
    c2 = f.transform_points(c1)
    area2 = 0.5 *abs((c2[0,0]*c2[1,1] - c2[1,0]*c2[0,1]) + (c2[1,0]*c2[2,1] - c2[2,0]*c2[1,1])+ (c2[2,0]*c2[3,1] - c2[3,0]*c2[2,1])+ (c2[3,0]*c2[0,1] - c2[0,0]*c2[3,1]))
    print area1, area2
    return c1,c2


def get_fringe(prob_nm, x_nd, y_md, block_lengths):
    try:
        for i in range(block_lengths[-1][0]): #stabilization points (not core or cross-section)
            j = np.argmax(prob_nm[-i-1])      # point with highest correspondence
            d1 = min([np.linalg.norm(x_nd[-i-1] - x_nd[c]) for c in range(block_lengths[0][0])]) #minimum dist to any core point
            if d1 < 0.015: #closer than intended radius of stabilization points
                prob_nm[-i-1,:] = 0
                prob_nm[:,-i-1] = 0
                prob_nm[j,:] = 0
                prob_nm[:,j] = 0
                # prob_nm = np.delete(prob_nm, [-i,j], axis=0)
                # prob_nm = np.delete(prob_nm, [-i,j], axis=1)
    except Exception as err:
        print "error in get_fringe"
        print err
        import IPython; IPython.embed()
    return prob_nm


def tps_rpm_bij_2d(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
                plotting = False, plot_cb = None, critical_points=0):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    supposed to find transform in 2d plane
    """
    import traceback

    x_nd = x_nd[:,:-1]
    y_md = y_md[:,:-1]
    if not np.isscalar(rot_reg):
        rot_reg = rot_reg[:-1]

    #adjust for duplicate points to prevent singular matrix errors
    for i in range(len(x_nd)):
        for j in range(len(x_nd)):
            if i!=j and np.all(x_nd[i]==x_nd[j]):
                x_nd[j][0] += 0.000001

    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g

    # r_N = None

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )

        for j in range(critical_points):
            prob_nm[j,:] = 0
            prob_nm[:,j] = 0
            prob_nm[j,j] = 1

        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 2e-1)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        
        try:
            f = fit_ThinPlateSpline2d(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
            g = fit_ThinPlateSpline2d(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)
        except:
            tb = traceback.format_exc()
            print "error in tps_rpm_bij"
            import IPython; IPython.embed() #import pdb; pdb.set_trace()
    f._cost = tps.tps_cost(f.lin_ag[:d,:d], f.trans_g[:d], f.w_ng[:,:d], f.x_na[:,:d], xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag[:d,:d], g.trans_g[:d], g.w_ng[:,:d], g.x_na[:,:d], ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()

    f.is_2d = True
    g.is_2d = True

    return f,g


def num_matches(ind,f):
    matches = 0
    for i in f.corr_nm:
        if np.argmax(i) == ind:
            matches += 1
    return matches

def get_matches(ind,f):
    matches = []
    for i in range(len(f.corr_nm)):
        if np.argmax(f.corr_nm[i]) == ind:
            matches.append(i)
    return matches

def assert_rigid(ptc1, ptc2, tol):
    if len(ptc1) != len(ptc2):
        return False
    else:
        lenptc = len(ptc1)
    for i in range(lenptc):
        for j in range(lenptc):
            if abs(np.linalg.norm(ptc1[i]-ptc1[j])-np.linalg.norm(ptc2[i]-ptc2[j])) > tol:
                print i, j, np.linalg.norm(ptc1[i]-ptc1[j]), np.linalg.norm(ptc2[i]-ptc2[j])
                return False
    return True

def assert_equal(ptc1, ptc2, tol):
    if len(ptc1) != len(ptc2):
        return False
    for i in range(len(ptc1)):
        if np.any(abs(ptc1[i]-ptc2[i]) > tol):
            print i, j, ptc1[i], ptc1[j]
            return False
    return True


def tps_reg_cost(f):
    K_nn = tps.tps_kernel_matrix(f.x_na)
    cost = 0
    for w in f.w_ng.T:
        cost += w.dot(K_nn.dot(w))
    return cost
    
def logmap(m):
    "http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29"
    theta = np.arccos(np.clip((np.trace(m) - 1)/2,-1,1))
    return (1/(2*np.sin(theta))) * np.array([[m[2,1] - m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]]]), theta


def balance_matrix3(prob_nm, max_iter, p, outlierfrac, r_N = None):
    
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m] = p
    prob_NM[n, :m] = p
    prob_NM[n, m] = p*np.sqrt(n*m)
    
    a_N = np.ones((n+1),'f4')
    a_N[n] = m*outlierfrac
    b_M = np.ones((m+1),'f4')
    b_M[m] = n*outlierfrac
    
    if r_N is None: r_N = np.ones(n+1,'f4')

    for _ in xrange(max_iter):
        c_M = b_M/r_N.dot(prob_NM)
        r_N = a_N/prob_NM.dot(c_M)

    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    
    return prob_NM[:n, :m], r_N, c_M

def balance_matrix(prob_nm, p, max_iter=20, ratio_err_tol=1e-3):
    n,m = prob_nm.shape
    pnoverm = (float(p)*float(n)/float(m))
    for _ in xrange(max_iter):
        colsums = pnoverm + prob_nm.sum(axis=0)        
        prob_nm /=  + colsums[None,:]
        rowsums = p + prob_nm.sum(axis=1)
        prob_nm /= rowsums[:,None]
        
        if ((rowsums-1).__abs__() < ratio_err_tol).all() and ((colsums-1).__abs__() < ratio_err_tol).all():
            break


    return prob_nm

def calc_correspondence_matrix(x_nd, y_md, r, p, max_iter=20):
    dist_nm = ssd.cdist(x_nd, y_md,'euclidean')
    
    
    prob_nm = np.exp(-dist_nm / r)
    # Seems to work better without **2
    # return balance_matrix(prob_nm, p=p, max_iter = max_iter, ratio_err_tol = ratio_err_tol)
    outlierfrac = 1e-1
    return balance_matrix3(prob_nm, max_iter, p, outlierfrac)


def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x


def fit_score(src, targ, dist_param):
    "how good of a partial match is src to targ"
    sqdists = ssd.cdist(src, targ,'sqeuclidean')
    return -np.exp(-sqdists/dist_param**2).sum()

def orthogonalize3_cross(mats_n33):
    "turns each matrix into a rotation"

    x_n3 = mats_n33[:,:,0]
    # y_n3 = mats_n33[:,:,1]
    z_n3 = mats_n33[:,:,2]

    znew_n3 = math_utils.normr(z_n3)
    ynew_n3 = math_utils.normr(np.cross(znew_n3, x_n3))
    xnew_n3 = math_utils.normr(np.cross(ynew_n3, znew_n3))

    return np.concatenate([xnew_n3[:,:,None], ynew_n3[:,:,None], znew_n3[:,:,None]],2)

def orthogonalize3_svd(x_k33):
    u_k33, _s_k3, v_k33 = svds.svds(x_k33)
    return (u_k33[:,:,:,None] * v_k33[:,None,:,:]).sum(axis=2)

def orthogonalize3_qr(_x_k33):
    raise NotImplementedError
