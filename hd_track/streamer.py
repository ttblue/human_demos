# A simple class to return a time-stamped list of transforms in a serialized manner.
from __future__ import division
import numpy as np

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
    
    def __init__(self, objs, ts, freq, favg, tstart=0):
        assert len(objs)==len(ts), "time-stamps and objects should have the same length"
        self.objs = objs
        
        self.ts   = np.array(ts)
        self.tmax = self.ts[-1]
        
        self.favg = favg
        self.dt   = 1./freq
        
        self.idx  = 0
        self.t    = tstart
    
    def __iter__(self):
        return self

    def next(self):
        if not self.idx < len(self.ts):
            raise StopIteration
        else:
            ttarg    = self.t + self.dt
            tidx     = self.idx

            while tidx < len(self.ts) and self.ts[tidx] < ttarg:
                tidx   += 1

            cands    = self.objs[self.idx:tidx]
            
            self.idx = tidx
            self.t  += self.dt
            
            return self.favg(cands) if cands else None
        
    def get_data(self):
        return (self.objs, self.ts)

            
            
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
    