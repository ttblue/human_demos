# A simple class to return a time-stamped list of transforms in a serialized manner.
from __future__ import division
import numpy as np
from hd_utils.colorize import colorize

class streamize():
    """
    A class that takes in a list of objects and their time-stamps.
    Also takes a function which returns the "average" of a list of objects.
 
    Returns a 'stream' of transforms indexed by a time-frequency:
    =============================================================
       On each call of 'next', it increments its time counter by
       1./freq and returns the average of all the objects b/w  t, t+1/f.
    
       If no object is present in [t,t+1/f], it returns None.

    It assumes, that time-stamps are sorted.
    Time-stamps are assumed to be numpy arrays of float.
    
    This class is iterable.
    """
    
    def __init__(self, objs, ts, freq, favg, tstart=None):
        assert len(objs)==len(ts), "time-stamps and objects should have the same length"
        self.objs = objs
        self.ts = np.array(ts)
    
        if len(ts) == 0:
            self.tmax = 0
        else:
            self.tmax = self.ts[-1]
        
        self.favg = favg
        self.dt   = 1./freq
        
        self.idx  = 0

        if tstart==None:
            if any(self.ts):
                tstart = np.min(self.ts) - self.dt
            else:
                tstart = -self.dt
        self.t    = tstart
        self.tstart = tstart

    
    def __iter__(self):
        return self

    def next(self):
        if not self.idx < len(self.ts):
            raise StopIteration
        else:
            ttarg    = self.t + self.dt
            tidx     = self.idx

            while tidx < len(self.ts) and self.ts[tidx] <= ttarg:
                tidx   += 1

            cands    = self.objs[self.idx:tidx]
            
            self.idx = tidx
            self.t  += self.dt
            
            return self.favg(cands) if cands else None
        
    def get_data(self):
        return (self.objs, self.ts)
    
    def reset(self):
        self.t   = self.tstart
        self.idx = 0

    def set_start_time(self, tstart):
        self.tstart = tstart
        self.reset()
        
    def get_start_time(self):
        return self.tstart

def time_shift_stream(strm, dT):
    """
    Time-shifts the stream by dT : a positive number delays the stream.
    Basically re-times the time-stamps and returns a new stream.
    """
    shift_stream = streamize(strm.objs, strm.ts+dT, 1./strm.dt, strm.favg, strm.tstart)
    return shift_stream 


def soft_next(stream):
    """
    Does not throw a stop-exception if a stream ends. Instead returns none.
    """
    ret = None
    try:
        ret = stream.next()
    except:
        pass
    return ret


def get_corresponding_data(strm1, strm2):
    """
    Returns two lists of same-length where the corresponding enteries occur at the same time-stamp.
    STRM1, STRM2 are the two streams for which the corresponding enteries are to be found.
    It also returns the indices at which data from both the streams is found.

    It assumes that the two streams start at the same time-scale.
    """
    dat1, dat2, inds = [],[],[]
    idx = 0
    while True:
        try:
            d1 = strm1.next()
            d2 = strm2.next()
            if d1!=None and d2!= None:
                dat1.append(d1)
                dat2.append(d2)
                inds.append(idx)
            idx += 1
        except:
            break

    N = len(dat1)
    print colorize("Found %s corresponding data points." % colorize(str(N), "red", True), "blue", True)
    
    return (inds, dat1, dat2)


if __name__ == '__main__':
    a  = [1,2,3,4,5,6]
    ts = [1,2,3,4,5,6]

    print "Testing\n========="
    print "objects = ", a
    print "times   = ", ts
    print 
    strm1 = streamize(a,ts, 2, np.mean)
    print "streaming with frequency = 2 : "
    for s in strm1:
        print "\t",s
    print
    strm2 = streamize(a,ts, 0.5, np.mean)
    print "streaming with frequency = 0.5 : "
    for s in strm2:
        print "\t",s
    