import openravepy as opr
import numpy as np
import time

import hd_utils.stl_utils as stlu

handles = []
env = opr.Environment()
env.Load('/home/sibi/sandbox/human_demos/hd_data/sensors/sensor_robot_2.00.xml')
sr = env.GetRobots()[0]
ss = env.GetSensors()[0]
ss.Configure(ss.ConfigureCommand.PowerOn)

# time.sleep(0.1)
# x1 = ss.GetSensorData(ss.Type.Laser).ranges
# 
# time.sleep(0.1)
# 
# tfm = np.eye(4)
# tfm[0:3,0:3] *= -1
# sr.SetTransform(tfm)
# x2 = ss.GetSensorData(ss.Type.Laser).ranges
# x2 = tfm.dot(np.r_[x2.T,np.atleast_2d(np.ones(x2.shape[0]))])[0:3,:].T
# 
# xx = np.vstack((x1,x2))
# colors = np.zeros_like(xx)
# colors[:,0] = 1
env.SetViewer('qtcoin')
check_points = [np.array([0,0,1.0])]#stlu.generate_sphere_points(1)

surface_points = np.empty((0,3))
for point in check_points:
    time.sleep(1)
    pos_points = stlu.get_points_from_position(sr, point, 2.00, check_valid=False)
    surface_points = np.vstack((surface_points, pos_points))
    handles.append(env.drawarrow(point, np.array([0,0,0])))
    

colors = np.zeros_like(surface_points)
colors[:,0] = 1

handles.append(env.plot3(points=surface_points,pointsize=2,colors=colors))