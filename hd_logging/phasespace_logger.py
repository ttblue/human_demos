import itertools
import os.path as osp, time

import OWL as owl
import hd_calib.phasespace as ph

log_freq = 5.0

def log_phasespace_markers (file_name=None):
    
    log_loc = "/home/sibi/sandbox/human_demos/hd_data/phasespace_logs"
    if file_name == None:
        base_name = "phasespace_log"    
        file_base = osp.join(log_loc, base_name)
        for suffix in itertools.chain("", (str(i) for i in itertools.count())):
            if not osp.isfile(file_base+suffix+'.log'):
                file_name = file_base+suffix+'.log'
                with open(file_name,"w") as fh: fh.write('')
                break
    else:
        file_name = osp.join(file_base, file_name)
        with open(file_name,"w") as fh: fh.write('')

    ph.turn_phasespace_on()
    start_time = time.time()

    while True:
        try:
            marker_pos = ph.get_marker_positions()
            time_stamp = time.time() - start_time
        except KeyboardInterrupt:
            break
        
        with open(file_name, "a") as fh:
            fh.write("- time_stamp: %f\n"%time_stamp)
            fh.write("  marker_positions: \n")
            for id in marker_pos:
                fh.write("   %i: "%id+str(marker_pos[id]) + "\n")

        try:
            #sleep for remainder of the time
            wait_time = time.time() - start_time - time_stamp
            time.sleep(max(1/log_freq-time.time()+start_time+time_stamp,0))
        except KeyboardInterrupt:
            break

    ph.turn_phasespace_off()
    print "Finished logging to file: "+file_name

