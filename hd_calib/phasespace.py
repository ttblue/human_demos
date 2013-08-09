import OWL as owl

MARKER_COUNT = 4
SERVER_NAME = "192.168.1.122"
INIT_FLAGS = 0

def turn_phasespace_on():
    if(owl.owlInit(SERVER_NAME, INIT_FLAGS) < 0):
        print "init error: ", owl.owlGetError()
        sys.exit(0)

    # create tracker 0
    tracker = 0
    owl.owlTrackeri(tracker, owl.OWL_CREATE, owl.OWL_POINT_TRACKER)
    # set markers

    for i in range(MARKER_COUNT):
        owl.owlMarkeri(owl.MARKER(tracker, i), owl.OWL_SET_LED, i)

    # activate tracker
    owl.owlTracker(tracker, owl.OWL_ENABLE)
    
    #return
    if(owl.owlGetStatus() == 0):
        owl.owl_print_error("error in point tracker setup", owl.owlGetError())
        sys.exit(0)
    owl.owlSetFloat(owl.OWL_FREQUENCY, owl.OWL_MAX_FREQUENCY)
    owl.owlSetInteger(owl.OWL_STREAMING, owl.OWL_ENABLE)

def turn_phasespace_off():
    owl.owlDone()
    
def get_marker_positions():
    
    marker_pos = {}
    while(1):
        markers = []
        n = owl.owlGetMarkers(markers, 50)
        err = owl.owlGetError()
        
        if (err != owl.OWL_NO_ERROR):
            break
        if(n==0): continue
        
        for marker in markers:
            #print "%d) %.2f %.2f %.2f" % (i, markers[i].x, markers[i].y, markers[i].z)
            marker_pos[marker.id] = [marker.x, marker.y, marker.z]
        break

    return marker_pos
