#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--source", type=str)
parser.add_argument("--target", type=str)

args = parser.parse_args()

import numpy as np
import pickle
import openravepy, trajoptpy
from hd_rapprentice import ropesim_floating, registration, plotting_openrave

handles = {}

class Globals:
	env = None
	sim = None
	viewer = None


def registration_cost_and_tfm(xyz0, xyz1, num_iters=30, critical_points=0, added_pts=0):
	scaled_xyz0, src_params = registration.unit_boxify(xyz0)
	scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
	f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=1, reg_final = .0001, rad_init = .1, rad_final = .0005,
				rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=num_iters, plotting=True)
	cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
	g = registration.unscale_tps_3d(g, targ_params, src_params)
	f = registration.unscale_tps_3d(f, src_params, targ_params)
	return (cost, f, g)


def plot_arrows(handles, xyz0, xyz1, max_num=100, step=10, size=0.0002, color=(1,0,0,1)):
	for i in range(max_num):
		handles.append(Globals.env.drawarrow(xyz0[i*step], xyz1[i*step], 0.0002, (1,0,0,1)))


def pickle_dump(xyz,filename):
    file1 = open(filename, 'wb')
    xyz = pickle.dump(xyz, file1)


def pickle_load(filename):
    file1 = open(filename, 'rb')
    xyz = pickle.load(file1)
    return xyz


def assert_equal(ptc1, ptc2, tol):
    if len(ptc1) != len(ptc2):
        return False
    for i in range(len(ptc1)):
        if np.any(abs(ptc1[i]-ptc2[i]) > tol):
            print i, ptc1[i], ptc2[i]
            return False
    return True


def plot_diff(pt_i,pt_j, old_xyz):
	orig_diff = np.array([(i/10.)*pt_i+(1-i/10.)*pt_j for i in range(11)])
	plot_orig_diff = Globals.env.plot3(orig_diff, 10, np.array([(1,0,1,1) for i in orig_diff]))
	tfmd_diff = f.transform_points(orig_diff); plot_tfmd_diff = Globals.env.plot3(tfmd_diff, 10, np.array([(1,0,1,1) for i in orig_diff]))
	plot_tfmd_strip = Globals.env.drawlinestrip(tfmd_diff,5,(1,1,0,1))


def main():
	np.set_printoptions(precision=5, suppress=True)
	Globals.env = openravepy.Environment()
	Globals.env.StopSimulation()
	Globals.sim = ropesim_floating.FloatingGripperSimulation(Globals.env)
	Globals.viewer = trajoptpy.GetViewer(Globals.env)

	if args.source and args.target:
		source_xyz = pickle_load(args.source)
		target_xyz = pickle_load(args.target)
	else:
		source_xyz = pickle_load("demo_xyz")
		target_xyz = pickle_load("sim_xyz")

	cost, f, g = registration_cost_and_tfm(source_xyz, target_xyz)


	plot_fake = Globals.env.plot3(source_xyz, 10, np.array([(0,1,0,1) for i in range(len(source_xyz))]))
	plot_fake2 = Globals.env.plot3(target_xyz, 10, np.array([(0,1,1,1) for i in range(len(target_xyz))]))
	plot_fake3 = Globals.env.plot3(f.transform_points(source_xyz), 11, np.array([(1,1,0,1) for i in range(len(source_xyz))]))
	handles = []
	handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, source_xyz.min(axis=0)-np.r_[0,0,.1], 
												source_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))
	Globals.viewer.Idle()
	import IPython; IPython.embed()

	# using only translation component of f
	plot_fake4 = Globals.env.plot3(source_xyz+f.trans_g, 11, np.array([(1,0,1,1) for i in range(len(source_xyz))]))

	# using translation and rotation components of f
	rigid_tfm = np.dot(source_xyz, f.lin_ag) + f.trans_g[None,:]
	plot_fake5 = Globals.env.plot3(rigid_tfm, 15, np.array([(0,0,1,1) for i in range(len(source_xyz))]))

	Globals.viewer.Step()
	import IPython; IPython.embed()

	#compute warp without scaling/boxify
	f2,g2 = registration.tps_rpm_bij(source_xyz, target_xyz, reg_init=1, reg_final = .0001, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=30)
	cost2 = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
	plot_fake6 = Globals.env.plot3(f2.transform_points(source_xyz), 15, np.array([(1,0,.5,1) for i in range(len(source_xyz))]))

	# using only translation component of f2
	plot_fake7 = Globals.env.plot3(source_xyz+f2.trans_g, 20, np.array([(2,.5,0,1) for i in range(len(source_xyz))]))

	# using translation and rotation components of f2
	rigid_tfm = np.dot(source_xyz, f2.lin_ag) + f2.trans_g[None,:]
	plot_fake8 = Globals.env.plot3(rigid_tfm, 15, np.array([(1,.1,.1,1) for i in range(len(source_xyz))]))

	Globals.viewer.Step()
	import IPython; IPython.embed()


if __name__ == "__main__":
    main()







