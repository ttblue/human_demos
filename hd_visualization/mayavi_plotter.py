import numpy as np
import mayavi_utils
from rapprentice.colorize import *
from mayavi import mlab
from multiprocessing import Process,Pipe
from threading import Lock
import cPickle
import time
from new_timer import MyTimer


def gen_custom_request(func_name, *args, **kwargs):
    """
    Returns a plotting request for custom functions.
    func_name should be in : {'points', 'lines'}
    """
    req = {'type':'custom', 'func':func_name, 'data':(args, kwargs)}
    return cPickle.dumps(req)


def gen_mlab_request(func, *args, **kwargs):
    """"
    Returns a plotting request for mlab function calls.
    """
    req = {'type':'mlab', 'func':gen_mlab_request.func_to_str[func], 'data':(args, kwargs)}
    return  cPickle.dumps(req)
gen_mlab_request.func_to_str = {v:k for k,v in mlab.__dict__.iteritems() if not k.startswith('_')}


class Plotter():
    """
    This class's check_and_process function is polled
    by the timer.
   
    A mayavi figure is created in a separate process and
    a timer is started in that same process.
   
    The timer periodically polls the check_and_process function
    which checks if any plotting request was sent to the process.
    If a request is found, the request is handled.
    """
    def __init__(self, in_pipe):
        self.request_pipe  = in_pipe
        self.plotting_funcs = {'lines': mayavi_utils.plot_lines,
                               'transform':mayavi_utils.plot_transform}

    def check_and_process(self):
        if self.request_pipe.poll():
            plot_request = self.request_pipe.recv()
            if plot_request:
                self.process_request(cPickle.loads(plot_request))

    def process_request(self, req):
        if req['type'] == 'custom':
            try:
                f = self.plotting_funcs[req['func']]        
            except KeyError:
                print colorize("No custom plotting function with name : %s. Ignoring plot request."%req['func'], "red")

        elif req['type'] == 'mlab':
            try:
                f = getattr(mlab, req['func'])
            except KeyError:
                print colorize("No mlab plotting function with name : %s. Ignoring plot request."%req['func'], "red")
        
        args, kwargs = req['data']
        f(*args, **kwargs)


@mlab.show
def create_mayavi(pipe):
    
    def send_key_callback(widget, event):
        "Send the key-presses to the process which created this mayavi viewer."
        pipe.send(cPickle.dumps(("KEY_DOWN", widget.GetKeyCode())))

    fig = mlab.figure()
    fig.scene.interactor.add_observer("KeyPressEvent", send_key_callback)
    time.sleep(1)

    mayavi_app = Plotter(pipe)

    from pyface.timer.api import Timer
    from mayavi.scripts import mayavi2

    timer = Timer(50, mayavi_app.check_and_process) 
    mayavi2.savedtimerbug = timer


class PlotterInit(object):
    """
    Initializes Mayavi in a new process.
    """
    def __init__(self):
        (self.pipe_mayavi, self.pipe_this) = Pipe()

        #process key-callbacks from mayavi scenes:        
        self.key_cb = {}
        self.lock = Lock()
        self.abc = list()
        from pyface.timer.api import Timer
        self.keycb_timer = MyTimer(0.05, self.__key_callback_server__)
        
        # start the mayavi process
        self.mayavi_process = Process(target=create_mayavi, args=(self.pipe_mayavi,))
        self.mayavi_process.start()

    def request(self, plot_request):
        self.pipe_this.send(plot_request)

    def register_key_callback(self, kbkey, f_cb):
        """
        NOTE : Currently callbacks can only be registered for SMALL keys 'a' --> 'z'
        """
        self.lock.acquire()
        self.key_cb[kbkey] = f_cb
        self.abc.append('1111')
        print self.abc
        print self.key_cb
        self.lock.release()
        

    def __key_callback_server__(self):
        if self.pipe_this.poll():
            key_press = self.pipe_this.recv()
            if key_press:
                event, key_char =  cPickle.loads(key_press)
                if event=="KEY_DOWN" and self.key_cb.has_key(key_char):
                    self.lock.acquire()
                    self.key_cb[key_char]()
                    self.lock.release()


if __name__=='__main__':

    # Example usage below:
    #=====================

    p = PlotterInit()
    parity = False

    while True:
        req = None
        color =  tuple(np.random.rand(1,3).tolist()[0])

        if parity:
            # example of a custom request to the plotter
            N = 5 # plot 5 sets of lines.
            line_points = [np.random.randn(2,3) for i in xrange(N)]
            req  = gen_custom_request('lines', lines=line_points, color=color, line_width=1, opacity=1)
            p.request(req)
        else:
            # example of how to request a mlab function to the plotter
            data  = np.random.randn(5,3)
            req   =  gen_mlab_request(mlab.points3d, data[:,0], data[:,1], data[:,2], color=color)
            p.request(req)

        parity = not parity
