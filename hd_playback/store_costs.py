import h5py
import numpy as np
#from rapprentice import registration

from joblib import Parallel, delayed
import cPickle as pickle
import argparse
import os.path as osp
import time

from hd_rapprentice import registration
from hd_utils import clouds
from hd_utils.defaults import demo_files_dir, similarity_costs_dir

np.set_printoptions(precision=6, suppress=True)

"""
Save reg params.

For each (i,j) seg pair:
1. Save TPS cost.
2. Save TPS unscaled cost.
3. Save Traj TPS cost.
4. Save transformed Traj TPS cost.
5. Save unscaled transformed Traj TPS cost.

Save TPS costs + reg params.
Save Transformed Traj cost + reg params.
Save Traj cost + reg params.
"""

tps_rot_reg = 1e-3
tps_n_iter = 30

traj_n = 10
traj_bend_c = 0.1
#traj_rot_c = [1e-3, 1e-3, 1e-3]
traj_rot_c = 1e-5
traj_scale_c = 0.1


def lerp (x, xp, fp, first=None):
    """
    Returns linearly interpolated n-d vector at specified times.
    """
    
    fp = np.asarray(fp)
    
    fp_interp = np.empty((len(x),0))
    for idx in range(fp.shape[1]):
        if first is None:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx])).T
        else:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx],left=first[idx])).T
        fp_interp = np.c_[fp_interp, interp_vals]
    
    return fp_interp


"""Doing min of what I need"""
def registration_cost(xyz0_p, xyz1_p):
    f,g = registration.tps_rpm_bij(xyz0_p[0], xyz1_p[0], rot_reg=tps_rot_reg, n_iter=tps_n_iter)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    f = registration.unscale_tps(f, xyz0_p[1], xyz1_p[1])
    g = registration.unscale_tps(g, xyz1_p[1], xyz0_p[1])

    return cost, f, g

def traj_cost(traj1, traj2, f, g):
    """
    Downsampled traj to have n points from start to end.
    """
    
    fo = registration.fit_ThinPlateSpline(traj1, traj2, traj_bend_c, traj_rot_c)
    go = registration.fit_ThinPlateSpline(traj2, traj1, traj_bend_c, traj_rot_c)
    cost = (registration.tps_reg_cost(fo)+registration.tps_reg_cost(go))/2
    
    if f is not None and g is not None:
        traj1_f = f.transform_points(traj1)
        traj2_g = g.transform_points(traj2)
        fn1 = registration.fit_ThinPlateSpline(traj1_f, traj2, traj_bend_c, traj_rot_c)
        gn1 = registration.fit_ThinPlateSpline(traj2, traj1_f, traj_bend_c, traj_rot_c)
        fn2 = registration.fit_ThinPlateSpline(traj1, traj2_g, traj_bend_c, traj_rot_c)
        gn2 = registration.fit_ThinPlateSpline(traj2_g, traj1, traj_bend_c, traj_rot_c)
        cost_fg = (registration.tps_reg_cost(fn1)+registration.tps_reg_cost(gn1)+
                   registration.tps_reg_cost(fn2) +registration.tps_reg_cost(gn2))/4
        return cost, cost_fg

    return cost



def all_costs(seg1, seg2, name):
    
    costs = {}
    #tps_c, f, _ = registration_cost(seg1[0], seg2[0], return_f=True)
    tpsc, f, g = registration_cost(seg1[0], seg2[0])
    lc, lfc = traj_cost(seg1[1], seg2[1], f, g)
    rc, rfc = traj_cost(seg1[2], seg2[2], f, g)

    costs['tps'] = tpsc
    costs['traj'] = {'l': lc,'r': rc}
    costs['traj_f'] = {'l': lfc,'r': rfc} 

    return name, costs

def get_name(demo_seg):
    demo,seg = demo_seg
    try:
        return 'd%i'%int(demo[4:])+'s%i'%int(seg[3:])
    except:
        return demo+'-'+seg


"""
Old, don't use.
"""
def save_costs(segs, keys, cost_file, num_segs=None):
    costs = {}
    costs['parameters'] = {}
    costs['parameters']['tps'] = {'tps_rot_reg':tps_rot_reg, 'tps_n_iter':tps_n_iter}
    costs['parameters']['traj'] = {'traj_n':traj_n, 'traj_bend_c':traj_bend_c,
                                   'traj_rot_c':traj_rot_c}#, 'traj_scale_c':traj_scale_c}
    
    costs['costs'] = {}
    
    if num_segs is None:
        num_segs = len(segs)

    for y in xrange(num_segs):
        new_seg = segs[y]
        name = get_name(keys[y])
        ts = time.time()
        print 'Segment '+name
        seg_costs = Parallel(n_jobs=-2,verbose=0)(delayed(all_costs)(new_seg, segs[i], get_name(keys[i])) for i in range(num_segs) if i != y )
        te = time.time()
        print "Time: %f"%(te-ts)
        costs[name] = {name_seg:cost for name_seg,cost in seg_costs}
        costs[name][name] = {c:0.0 for c in ['tps', 'traj','traj_f']}
        
    with open(cost_file,'w') as fh: pickle.dump(costs, fh)

def save_costs_symmetric(segs, keys, cost_file, num_segs=None):
    data = {}
    data['parameters'] = {}
    data['parameters']['tps'] = {'tps_rot_reg':1e-3, 'tps_n_iter':50}
    data['parameters']['traj'] = {'traj_n':20, 'traj_bend_c':0.05,
                                   'traj_rot_c':[1e-3, 1e-3, 1e-3], 'traj_scale_c':0.1}
    
    name_keys = {i:get_name(keys[i]) for i in keys}
    costs = {name_keys[i]:{} for i in name_keys}
    
    if num_segs is None:
        num_segs = len(segs)

    for y in xrange(num_segs):
        new_seg = segs[y]
        name = name_keys[y]
        ts = time.time()
        print 'Segment '+name_keys[y]
        seg_costs = Parallel(n_jobs=4,verbose=51)(delayed(all_costs)(new_seg, segs[i], name_keys[i]) for i in range(y+1, num_segs))
        te = time.time()
        print "Time: %f"%(te-ts)

        costs[name][name] = {'tps':0.0, 'traj':{'l':0.0,'r':0.0}, 'traj':{'l':0.0,'r':0.0}}
        for name_seg, cost in seg_costs:
            costs[name][name_seg] = cost
        
    data['costs'] = costs
    with open(cost_file,'w') as fh: pickle.dump(data, fh)


def extract_segs(demofile, num_segs=None):
    seg_num = 0
    leaf_size = 0.045
    keys = {}
    segs = []
    for demo_name in demofile:
        if demo_name != "ar_demo":
            for seg_name in demofile[demo_name]:
                if seg_name != 'done':
                    keys[seg_num] = (demo_name, seg_name)
                    seg = demofile[demo_name][seg_name]
                    pc = clouds.downsample(np.asarray(seg['cloud_xyz']), leaf_size)
                    pc, params = registration.unit_boxify(pc)
                    ltraj = np.asarray(seg['l']['tfms_s'])[:,0:3,3]
                    ltraj = lerp(np.linspace(0,ltraj.shape[0],traj_n), range(ltraj.shape[0]), ltraj)
                    rtraj = np.asarray(seg['r']['tfms_s'])[:,0:3,3]
                    rtraj = lerp(np.linspace(0,rtraj.shape[0],traj_n), range(rtraj.shape[0]), rtraj)
                    
                    segs.append(((pc, params), ltraj, rtraj))
                    seg_num += 1
                    if num_segs is not None and seg_num >= num_segs: return keys, segs
    
    return keys, segs
    


def main(demo_type, num_segs=None):
    demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
    
    iden = ''
    if num_segs is not None:
        iden = str(num_segs)
    cost_file = osp.join(demo_files_dir, demo_type)+demo_type+iden+'.costs'
    
    keys, segs = extract_segs(demofile, num_segs)
    save_costs_symmetric(segs, keys, cost_file, num_segs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type",help="Demo type.", type=str)
    parser.add_argument("--num_segs",help="Number of segs.", type=int, default = -1)

    args = parser.parse_args()

    ns = args.num_segs
    if ns < 0: ns = None

    main(args.demo_type, ns)
