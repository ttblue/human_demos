import trajoptpy, openravepy


def animate_traj(traj, base_hmats, robot, pause=True, step_viewer=True, restore=True, callback=None):
    """make sure to set active DOFs beforehand"""
    if restore: _saver = openravepy.RobotStateSaver(robot)
    if step_viewer or pause: viewer = trajoptpy.GetViewer(robot.GetEnv())
    for (i,dofs) in enumerate(traj):
        if i%10 == 0: print "step %i/%i"%(i+1,len(traj))
        if callback is not None: callback(i)
        robot.SetActiveDOFValues(dofs)
        if base_hmats != None:
            robot.SetTransform(base_hmats[i])
        if pause: viewer.Idle()
        elif step_viewer and i%50 == 0: viewer.Step()
        
def animate_floating_traj(lhmats, rhmats, sim, pause=True, step_viewer=True, callback=None,step=5):
    assert len(lhmats)==len(rhmats), "I don't know how to animate trajectory with different lengths"
    if step_viewer or pause: viewer = trajoptpy.GetViewer(sim.env)
    for i in xrange(len(lhmats)):
        if callback is not None: callback(i)
        sim.grippers['r'].set_toolframe_transform(rhmats[i])
        sim.grippers['l'].set_toolframe_transform(lhmats[i])
        if pause: viewer.Idle()
        elif step_viewer and not i%step: viewer.Step()

def animate_floating_traj_angs(lhmats, rhmats, ljoints, rjoints, sim, pause=True, step_viewer=True, callback=None,step=5):
    assert len(lhmats)==len(rhmats), "I don't know how to animate trajectory with different lengths"
    if step_viewer or pause: viewer = trajoptpy.GetViewer(sim.env)
    for i in xrange(len(lhmats)):
        if callback is not None: callback(i)
        sim.grippers['l'].set_toolframe_transform(lhmats[i])
        sim.grippers['r'].set_toolframe_transform(rhmats[i])
        if ljoints!=None and rjoints!=None:
            sim.grippers['l'].set_gripper_joint_value(ljoints[i])
            #print "l: ", ljoints[i]
            sim.grippers['r'].set_gripper_joint_value(rjoints[i])
            #print "r: ", rjoints[i]
        if pause: viewer.Idle()
        elif step_viewer and not i%step: viewer.Step()