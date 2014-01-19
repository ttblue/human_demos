import openravepy,trajoptpy, numpy as np, json

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, end_pose_constraint=False):
        
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    arm_inds  = robot.GetManipulator(manip_name).GetArmIndices()

    ee_linkname = ee_link.GetName()
    
    init_traj = old_traj.copy()
    #init_traj[0] = robot.GetDOFValues(arm_inds)
    pose = openravepy.poseFromMatrix(new_hmats[-1])

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
    if end_pose_constraint:
        request['constraints'] += [
             {"type":"pose",
                "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
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
