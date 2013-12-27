import openravepy,trajoptpy, numpy as np, json


def inverse_kinematics(robot, manip_name, ee_hmat):
    init_guess = np.zeros(7)
    
    xyz_target = ee_hmat[:3, 3]
    quat_target = openravepy.quatFromRotationMatrix(ee_hmat[:3, :3])
    
    
    request = {
      "basic_info" : {
        "n_steps" : 10,
        "manip" : manip_name, # see below for valid values
        "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
      },
      "costs" : [
      {
        "type" : "joint_vel", # joint-space velocity cost
        "params": {"coeffs" : [1]} # a list of length one is automatically expanded to a list of length n_dofs
      },
      {
        "type" : "collision",
        "name" :"cont_coll", # shorten name so printed table will be prettier
        "params" : {
          "continuous" : True,
          "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
          "dist_pen" : [0.025] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
        }
      }
      ],
      "constraints" : [
      # BEGIN pose_constraint
      {
        "type" : "pose", 
        "params" : {"xyz" : xyz_target.tolist(), 
                    "wxyz" : quat_target.tolist(), 
                    "link": "r_gripper_tool_frame",
                    "timestep" : 9
                    }
                     
      }
      # END pose_constraint
      ],
      # BEGIN init
      "init_info" : {
          "type" : "straight_line", # straight line in joint space.
          "endpoint" : init_guess.tolist() # need to convert numpy array to list
      }
      # END init
    }
        

    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
    result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()
    
    return traj[-1]         