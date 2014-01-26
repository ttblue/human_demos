import openravepy, trajoptpy, numpy as np, json
import hd_utils.math_utils as mu
from hd_utils.colorize import redprint, blueprint


def mat_to_base_pose(mat):
    pose = openravepy.poseFromMatrix(mat)
    x = pose[4]
    y = pose[5]
    rot = openravepy.axisAngleFromRotationMatrix(mat)[2]
    return x, y, rot

def base_pose_to_mat(pose):
    x, y, rot = pose
    q = openravepy.quatFromAxisAngle((0, 0, rot)).tolist()
    pos = [x, y, 0]
    matrix = openravepy.matrixFromPose(q + pos)
    return matrix


def find_closest_point(points, query):
    dists = [np.linalg.norm(points[i, :] - query) for i in range(points.shape[0])]
    
    idx = np.argmin(dists)
    
    return points[idx, :]
    

  
def get_environment_limits(env, robot=None):
    """Calculates the limits of an environment, If robot is not None then then
    limits are shrunk by the size of the robot.

    Return:
    envmin, envmax: two 1x3 arrays with the environment minimun and maximum
    extension.
    """

    if robot is not None:
        abrobot = robot.ComputeAABB()
        min_r = abrobot.extents()
        max_r = -abrobot.extents()
    else:
        min_r = max_r = 0

    with env:
        envmin = []
        envmax = []
        for b in env.GetBodies():
            ab = b.ComputeAABB()
            envmin.append(ab.pos()-ab.extents())
            envmax.append(ab.pos()+ab.extents())
        envmin = np.min(np.array(envmin),0) + min_r
        envmax = np.max(np.array(envmax),0) + max_r

    return envmin, envmax


def traj_to_bodypart_traj(traj, bodypart_keys):
    bodypart_traj = {}
    offset = 0
    if 'rightarm' in bodypart_keys:
        bodypart_traj['rightarm'] = traj[:, range(offset, offset+7)]
        offset += 7
    if 'leftarm' in bodypart_keys:
        bodypart_traj['leftarm'] = traj[:, range(offset, offset+7)]
        offset += 7
    if 'base' in bodypart_keys:
        bodypart_traj['base'] = traj[:, range(offset, offset+3)]
        offset += 3
    return bodypart_traj


def get_traj_collisions(bodypart_traj, robot, n = 100):
    
    traj = []
    
    for group in bodypart_traj:
        traj.append(bodypart_traj[group])
    
    traj_up = mu.interp2d(np.linspace(0,1,n), np.linspace(0,1,len(traj)), traj)
    
    env = robot.GetEnv()
    col_times = []
    for (i,row) in enumerate(traj_up):
        robot.SetActiveDOFValues(row)
        col_now = env.CheckCollision(robot)
        if col_now: 
            col_times.append(i)
    return col_times

def generate_traj(robot, env, costs, constraints, n_steps, init_traj, collisionfree):
    init_info = {
                 "type": "given_traj",
                 "data": init_traj
                 }
    
    
    request = {
               "basic_info": {
                              "n_steps": n_steps if (n_steps is None) else n_steps,
                              "manip": "active",
                              "start_fixed": False  # i.e., DOF values at first timestep are fixed based on current robot state
                              },
               "costs": costs,
               "constraints": constraints,
               "init_info": init_info
               }
    
    request['costs'] += [{
                          "type": "collision",
                          "name": "cont_col_free",
                          "params": {
                                     "coeffs": [50],
                                     "dist_pen": [0.05]
                                     }
                          }, 
                         {
                          "type": "collision",
                          "name": "col",
                          "params": {
                                     "continuous": False,
                                     "coeffs": [20],
                                     "dist_pen": [0.02]
                                     }
                          }]
    
    s = json.dumps(request)
    
    
    prob = trajoptpy.ConstructProblem(s, env)

    
    result = trajoptpy.OptimizeProblem(prob)
    traj = result.GetTraj()
    total_cost = sum(cost[1] for cost in result.GetCosts())
    return traj, total_cost


def is_fake_motion(hmats, thresh):
    xyz_traj = [hmat[:3,3] for hmat in hmats]
    xyz_traj = np.asarray(xyz_traj)
    
    xyz_min = np.amin(xyz_traj, 0)
    xyz_max = np.amax(xyz_traj, 0)
    
    #print np.linalg.norm(xyz_min - xyz_max)
    
    return np.linalg.norm(xyz_min - xyz_max) < thresh

def plan_fullbody(robot, env, new_hmats_by_bodypart, old_traj_by_bodypart,
                    end_pose_constraints,
                    rope_cloud=None,
                    rope_constraint_thresh=.01,
                    allow_base=True,
                    init_data=None):
    
    collision_free=True
    for part_name in new_hmats_by_bodypart:
        n_steps = len(new_hmats_by_bodypart[part_name])
        break
    
    state = np.random.RandomState(None)

    init_dofs = []
    if 'base' in old_traj_by_bodypart.keys():
        if 'rightarm' in old_traj_by_bodypart.keys():
            init_dofs += old_traj_by_bodypart['rightarm'].tolist()
        else:
            init_dofs += robot.GetDOFValues(robot.GetManipulator('rightarm').GetArmIndices()).tolist()
        if 'leftarm' in old_traj_by_bodypart.keys():
            init_dofs += old_traj_by_bodypart['leftarm'].tolist()
        else:
            init_dofs += robot.GetDOFValues(robot.GetManipulator('leftarm').GetArmIndices()).tolist()
            
        init_dofs += old_traj_by_bodypart['base']
    else:
        arm_dofs = []
        if 'rightarm' in old_traj_by_bodypart.keys():
            arm_dofs = old_traj_by_bodypart['rightarm']
        if 'leftarm' in old_traj_by_bodypart.keys():
            arm_dofs = np.c_[arm_dofs, old_traj_by_bodypart['leftarm']]
        init_dofs = arm_dofs

        if allow_base:
            x, y, rot = mat_to_base_pose(robot.GetTransform())
            base_dofs = np.c_[x,y,rot]
            init_dofs = np.c_[init_dofs, np.repeat(base_dofs, init_dofs.shape[0], axis=0)]

        init_dofs = init_dofs.tolist()
            


            
    bodyparts = new_hmats_by_bodypart.keys()
    if 'base' in bodyparts: #problem
        bodyparts += ['rightarm', 'leftarm']
    elif allow_base:
        bodyparts.append('base')

     
    costs= []
    constraints = []
    if 'base' in bodyparts:
        if old_traj_by_bodypart is None:
            raise Exception("Base motion planning must have init_dofs")
        
        n_dofs = len(init_dofs[0])
        cost_coeffs = n_dofs * [5]
        cost_coeffs[-1] = 500
        cost_coeffs[-2] = 500
        cost_coeffs[-3] = 500
        
        
        joint_vel_cost = {
                          "type": "joint_vel",
                          "params": {"coeffs": cost_coeffs}
                          }
        
        costs.append(joint_vel_cost)
        
    else:
        joint_vel_cost = {
                          "type": "joint_vel",  # joint-space velocity cost
                          "params": {"coeffs": [5]}  # a list of length one is automatically expanded to a list of length n_dofs
                          }
        costs.append(joint_vel_cost)


    for manip, poses in new_hmats_by_bodypart.items():
        if manip == 'rightarm':
            link = 'r_gripper_tool_frame'
        elif manip == 'leftarm':
            link = 'l_gripper_tool_frame'
        elif manip == 'base':
            link = 'base_footprint'    
        
        if manip in ['rightarm', 'leftarm']: 
            if not end_pose_constraints[manip] and is_fake_motion(poses, 0.1):
                continue
            
            
        if manip in ['rightarm', 'leftarm']:
            if end_pose_constraints[manip] or not is_fake_motion(poses, 0.1):
                end_pose = openravepy.poseFromMatrix(poses[-1])
                
                if rope_cloud != None:
                    closest_point = find_closest_point(rope_cloud, end_pose[4:7])
                    dist = np.linalg.norm(end_pose[4:7] - closest_point)
                    if dist > rope_constraint_thresh and dist < 0.2:
                        end_pose[4:7] = closest_point
                        redprint("grasp hack is active, dist = %f"% dist)
                
                constraints.append({"type":"pose",
                                    "params":{
                                              "xyz":end_pose[4:7].tolist(),
                                              "wxyz":end_pose[0:4].tolist(),
                                              "link":link,
                                              "pos_coeffs":[10,10,10],
                                              "rot_coeffs":[10,10,10]}})
                
        
        poses = [openravepy.poseFromMatrix(hmat) for hmat in poses]
    
        for t, pose in enumerate(poses):
            costs.append({
                             "type": "pose",
                             "params": {"xyz": pose[4:].tolist(),
                                        "wxyz": pose[:4].tolist(),
                                        "link": link,
                                        "pos_coeffs": [20, 20, 20],
                                        "rot_coeffs": [20, 20, 20],
                                        "timestep": t}
                             })
            

    traj, total_cost = generate_traj(robot, env, costs, constraints, n_steps, init_dofs, collision_free)
    bodypart_traj = traj_to_bodypart_traj(traj, bodyparts)

    # Do multi init if base planning
    if 'base' in bodypart_traj and not allow_base:
        waypoint_step = (n_steps - 1) // 2
        env_min, env_max = get_environment_limits(env, robot)
        current_dofs = robot.GetActiveDOFValues().tolist()
        waypoint_dofs = list(current_dofs)
        
        MAX_INITS = 10
        n = 0
        while True:
            collisions = get_traj_collisions(bodypart_traj, robot)
            if collisions == []:
                break

        n += 1
        if n > MAX_INITS:
            print "Max base multi init limit hit!"
            return bodypart_traj, total_cost  # return arbitrary traj with collisions

        print "Base planning failed! Trying random init. Iteration {} of {}".format(n, MAX_INITS)

        # randomly sample x, y within env limits + some padding to accommodate robot
        padding = 5.0
        env_min_x = env_min[0] - padding
        env_min_y = env_min[1] - padding
        env_max_x = env_max[0] + padding
        env_max_y = env_max[1] + padding

        waypoint_x = state.uniform(env_min_x, env_max_x)
        waypoint_y = state.uniform(env_min_y, env_max_y)

        waypoint_dofs[-3] = waypoint_x
        waypoint_dofs[-2] = waypoint_y

        init_data = np.empty((n_steps, robot.GetActiveDOF()))
        init_data[:waypoint_step+1] = mu.linspace2d(current_dofs, waypoint_dofs, waypoint_step+1)
        init_data[waypoint_step:] = mu.linspace2d(waypoint_dofs, init_dofs, n_steps - waypoint_step)

        traj, total_cost = generate_traj(robot, env, costs, constraints, n_steps, collision_free,
                                         init_dofs, init_data)
        bodypart_traj = traj_to_bodypart_traj(traj, bodyparts)

    return bodypart_traj, total_cost         
        
            
            
        
            

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, rope_cloud=None, rope_constraint_thresh=.01, end_pose_constraint=False):
        
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    arm_inds  = robot.GetManipulator(manip_name).GetArmIndices()

    ee_linkname = ee_link.GetName()
    
    init_traj = old_traj.copy()
    #init_traj[0] = robot.GetDOFValues(arm_inds)
    end_pose = openravepy.poseFromMatrix(new_hmats[-1])

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : False
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [n_steps/5.]}
        },
        {
            "type" : "collision",
            "params" : {"coeffs" : [1],"dist_pen" : [0.01]}
        }                
        ],
        "constraints" : [ 
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj]
        }
    }
    
    request['costs'] += [{
        "type": "collision",
        "name": "cont_coll",
        "params": {
          "coeffs": [50],
          "dist_pen": [0.05]
        }
      }, {
        "type": "collision",
        "name": "col",
        "params": {
          "continuous": False,
          "coeffs": [20],
          "dist_pen": [0.02]
        }
      }]
    
    
    #impose that the robot goes to final ee tfm at last ts
    #the constraint works only when the arm is the 'grasp' arm; otherwise only cost is added
    if end_pose_constraint:# or not is_fake_motion(new_hmats, 0.1):
        
        # hack to avoid missing grasp
        if rope_cloud != None:
            closest_point = find_closest_point(rope_cloud, end_pose[4:7])
            dist = np.linalg.norm(end_pose[4:7] - closest_point)
            if dist > rope_constraint_thresh and dist < 0.05:
                end_pose[4:7] = closest_point
                redprint("grasp hack is active, dist = %f"% dist)
            else:
                blueprint("grasp hack is inactive, dist = %f"% dist)
            #raw_input()

        
        request['constraints'] += [
             {"type":"pose",
                "params":{
                "xyz":end_pose[4:7].tolist(),
                "wxyz":end_pose[0:4].tolist(),
                "link":ee_linkname,
                "pos_coeffs":[10,10,10],
                "rot_coeffs":[10,10,10]}}]

        
    poses = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats]
        
    for (i_step,pose) in enumerate(poses):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":ee_linkname,
                "timestep":i_step,
                "pos_coeffs":[10,10,10],
                "rot_coeffs":[10,10,10]
             }
            })


    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
    result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()    
        
    saver = openravepy.RobotStateSaver(robot)
    pos_errs = []    
    with saver:
        for i_step in xrange(1,n_steps):
            row = traj[i_step]
            robot.SetDOFValues(row, arm_inds)
            tf = ee_link.GetTransform()
            pos = tf[:3,3]
            pos_err = np.linalg.norm(poses[i_step][4:7] - pos)
            pos_errs.append(pos_err)
    pos_errs = np.array(pos_errs)
    print "planned trajectory for %s. max position error: %.3f. all position errors: %s"%(manip_name, pos_errs.max(), pos_errs)
            
    return traj         
