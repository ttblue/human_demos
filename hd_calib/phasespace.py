from OWL import *
import sys

MARKER_COUNT = 4
SERVER_NAME = "192.168.1.143"
INIT_FLAGS = 0

def owl_print_error(s, n):
    """Print OWL error."""
    if(n < 0): print "%s: %d" % (s, n)
    elif(n == OWL_NO_ERROR): print "%s: No Error" % s
    elif(n == OWL_INVALID_VALUE): print "%s: Invalid Value" % s
    elif(n == OWL_INVALID_ENUM): print "%s: Invalid Enum" % s
    elif(n == OWL_INVALID_OPERATION): print "%s: Invalid Operation" % s
    else: print "%s: 0x%x" % (s, n)


def turn_phasespace_on():
    if(owlInit(SERVER_NAME, INIT_FLAGS) < 0):
        print "init error: ", owlGetError()
        sys.exit(0)

    # create tracker 0
    tracker = 0
    owlTrackeri(tracker, OWL_CREATE, OWL_POINT_TRACKER)
    # set markers

    for i in range(MARKER_COUNT):
        owlMarkeri(MARKER(tracker, i), OWL_SET_LED, i)

    # activate tracker
    owlTracker(tracker, OWL_ENABLE)
    
    #return
    if(owlGetStatus() == 0):
        owl_print_error("error in point tracker setup", owlGetError())
        sys.exit(0)
    owlSetFloat(OWL_FREQUENCY, OWL_MAX_FREQUENCY)
    owlSetInteger(OWL_STREAMING, OWL_ENABLE)

def turn_phasespace_off():
    owlDone()
    
def get_marker_positions(print_shit=False):
    SCALE = 1000.0
    marker_pos = {}
    while(1):
        markers = []
        n = owlGetMarkers(markers, 32)
        
        err = owlGetError()
        if (err != OWL_NO_ERROR):
            owl_print_error("error", err)
            break
    
        if(n==0): continue
        if(n<0):
            print "Something is wrong"
            break
        
        if print_shit:
            print "---------------------"
            print markers
            print "---------------------"
        for marker in markers:
            #print "%d) %.2f %.2f %.2f" % (i, markers[i].x, markers[i].y, markers[i].z)
            if marker.cond > 0:
                x,y,z = marker.x, marker.y, marker.z
                
                x = x - int(x/100)
                y = y - int(y/100)
                z = z - int(z/100)
                
                marker_pos[marker.id] = [x*1/SCALE, y*1/SCALE, z*1/SCALE]


        break

    return marker_pos
