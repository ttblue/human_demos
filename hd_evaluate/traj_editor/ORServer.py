from openravepy import *
from  RopePR2Viz import RopePR2Viz
import cPickle

class ORServer(object):
    '''
    Server to receive [and send] commands from [to] Qt GUI.  
    '''
    
    def __del__(self):
        pass
    
    def __init__(self, pipe):
        self.pipe    = pipe
        self.running = True
        self.viz     = RopePR2Viz()
        self.plot_handle = None
        self.__run__()

    def __run__(self):
        while(self.running):
            functionName,args = self.pipe.recv()
            self.executeFunction(functionName, args)

    def Stop(self):
        self.running = False
        return None,"Stopping!!!"


    def executeFunction(self,name,args):        
        rValue = None
        rMessage = "Function with " + name + " not available"

        if name in dir(self):
            if(args is None):
                rValue,rMessage = getattr(self,name)()
            else:
                rValue,rMessage = getattr(self,name)(args)
        return rValue,rMessage


    def StartViewer(self):
        try:
            self.viz.env.SetViewer('qtcoin')
            return True,None
        except:
            pass
        return None,"OpenRave environment not up!"


    def SetRobotPose(self,pose_dat):
        dofs, tfm = cPickle.loads(pose_dat)
        self.viz.set_robot_pose(dofs, tfm)       
        return True, "pose set."


    def UpdateRope(self,control_pts):
        control_pts = cPickle.loads(control_pts)
        self.viz.update_rope(control_pts)       
        return True, "rope set."

    def PlotPoints(self, pts):
        pts = cPickle.loads(pts)
        assert pts.ndim==2 and pts.shape[1]==3, "ORServer : PlotPoints, unknown point-data."
        with self.viz.env:
            self.plot_handle = self.viz.env.plot3(points=pts, pointsize=5.0 )
        return True, "Points done."

