import os, os.path as osp
import time        


import openravepy as opr
import numpy as np, numpy.linalg as nlg

from colorize import redprint

handles = []

def get_sensor_robot(env, max_range=5.0):
    """
    Creates an xml file for a sensor and loads the data.
    If one already exists (naming convention is strict for xml file and robot name), 
    loads that one.
    
    Arguments:
    @env - openrave environment
    @max_range - maximum range of sensor
    Return:
    @sensor_robot - the robot loaded. 
                    Does not return sensor because can only set robot transform.
    
    Not quite sure what the minimum range of sensor is.
    """
    robot_name = 'sensor_robot_%.2f'%max_range
    sensor_file = osp.join(os.getenv('HD_DATA_DIR'),'sensors',robot_name+'.xml')
    if not osp.isfile(sensor_file):
        fo = open(sensor_file,'w')
        fo.write("<!-- Sensor robot of max_range %.2f-->\n"%max_range)
        fo.write("<Robot name=\"%s\">\n"%robot_name)
        fo.write("  <Kinbody>\n")
        fo.write("    <Body name=\"Base\" type=\"dynamic\">\n")
        fo.write("      <Translation>0.0 0.0 0.0</Translation>\n")
        fo.write("    </Body>\n")
        fo.write("  </Kinbody>\n")
        fo.write("  <AttachedSensor name=\"LIDAR\">\n")
        fo.write("    <link>Base</link>\n")
        fo.write("    <translation>0.0 0.0 0.0</translation>\n")
        fo.write("    <rotationaxis>0 0 1 0</rotationaxis>\n")
        fo.write("    <sensor type=\"BaseFlashLidar3D\">\n")
        fo.write("      <maxrange>%.2f</maxrange>\n"%max_range)
        fo.write("      <scantime>0.01</scantime>\n")
        fo.write("      <KK>32 24 32 24</KK>\n")
        fo.write("      <width>64</width>\n")
        fo.write("      <height>48</height>\n")
        fo.write("      <color>1 1 0</color>\n")
        fo.write("    </sensor>\n")
        fo.write("  </AttachedSensor>\n")
        fo.write("</Robot>")
        fo.close()
    
    sensor_robot = env.GetRobot(robot_name)     
    if sensor_robot is None:
        env.Load(sensor_file)
        sensor_robot = env.GetRobot(robot_name)
    
    sensor = sensor_robot.GetAttachedSensor('LIDAR').GetSensor()
    sensor.Configure(sensor.ConfigureCommand.PowerOn)
    
    return sensor_robot

def get_points_from_position(sensor_robot, position, max_range, check_valid=True):
    """
    Returns points from the frame of the object, as seen from some particular position.
    """
    
    # The sensor points in +z direction.
    position = position.astype('f')
    zvec = -position/nlg.norm(position)
    xvec = np.array([0.0,0.0,0.0])
    if zvec[0]==0:
        xvec[0] = 1.0
    else:
        xvec[0] = -zvec[1]
        xvec[1] = zvec[0]
        xvec /= nlg.norm(xvec)
    yvec = np.cross(zvec,xvec)
    
    # Create appropriate transform
    tfm = np.r_[np.c_[xvec,yvec,zvec,position],[[0,0,0,1]]]
    print tfm
    sensor_robot.SetTransform(tfm)
    
    # Get latest sensor data
    sensor = sensor_robot.GetAttachedSensors()[0].GetSensor()
    olddata = sensor.GetSensorData(sensor.Type.Laser)
    while True:
        data = sensor.GetSensorData(sensor.Type.Laser)
        if data.stamp != olddata.stamp:
            break
        time.sleep(0.1)
    points = data.ranges
    
    # Take only valid points (those not at max_range)
    if check_valid:
        valid_inds = abs((points**2).sum(axis=1)-max_range**2)>1e-3
        points = points[valid_inds,:]
    
    # Transform points to correct frame and return
    # tfm_points = np.r_[points.T,np.atleast_2d(np.ones(points.shape[0]))]
    # org_points = tfm.dot(tfm_points)
    return points + position
    
def generate_sphere_points(radius):
    """
    Generates six points on sphere, at +/- points along each axis.
    @radius - radius of sphere
    """
    points = []
    for i in range(3):
        point1 = np.array([0,0,0])
        point1[i] = radius
        point2 = np.array([0,0,0])
        point2[i] = -radius
        points.append(point1)
        points.append(point2)
    
    return points

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))        

def get_surface_point_cloud(object_files, env=None):
    """
    Gets points on the surface, as sampled from different view points
    """
    # Create env if None
    if env is None:
        env = opr.Environment()
    
    # Load file if possible
    radii = []
    for file in object_files:
        check = env.Load(file)
        if not check:
            redprint("Object file %s not found!"%file)
            return
        # Move object to origin
        # Compute AABB to find radius of sphere to check
        obj = env.GetBodies()[-1]
        obj.SetTransform(np.eye(4))
        aabb = obj.ComputeAABB()
        corner_points = []
        p = aabb.pos()
        ext = aabb.extents()
        corner_points.extend([p+ext, p-ext])
        for i in range(3):
            e = ext.copy()
            e[i] *= -1
            corner_points.extend([p+e, p-e])
        corner_points = np.asarray(corner_points)
        radii.append(np.sqrt(np.max((corner_points**2).sum(axis=1))))
        print radii
    
    # Add a little offset to radius
    min_thresh = max(radii)*0.5 
    radius = max(radii) + min_thresh
    # Max range for sensor - radius + diagonal distance + offset
    max_range = np.ceil(radius + max(radii)*0.25 + nlg.norm(ext)*2)

    # Create robot for sensor measurements
    sensor_robot = get_sensor_robot(env, max_range)
    
    # Points from which to find point clouds
    check_points = generate_sphere_points(radius)
    
    surface_points = np.empty((0,3))
    for point in check_points:
        time.sleep(0.1)
        pos_points = get_points_from_position(sensor_robot, point, max_range)
        surface_points = np.vstack((surface_points, pos_points))
    
    return unique_rows(surface_points)
    
def change_visibility(env, visible=False):
    for body in env.GetBodies():
        body.SetVisible(visible)

    
if __name__=='__main__':
    from os import listdir
    from os.path import isfile, join
     
    mypath = '/home/sibi/sandbox/pr2-lfd/assemblies/STLs/'
    object_files = [ join(mypath,f) for f in listdir(mypath) if isfile(join(mypath,f)) and "5Degree" in f]
    env = opr.Environment()
#     for file in object_files:
#         check = env.Load(file)
#      
    env.SetViewer('qtcoin')
    points = get_surface_point_cloud(object_files,env)#['robots/pr2-beta-static.zae'], env)
    colors = np.empty_like(points)
    colors[:] = 0
    colors[:,0] = 1
    handles.append(env.plot3(points=points, pointsize=2, colors=colors))

    

#     r = env.GetRobot('pr2')
#     tfm = np.eye(4)
#     tfm[0,3] = 2
    #r.SetTransform(tfm)
