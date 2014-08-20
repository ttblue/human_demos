from PIL import Image, ImageOps, ImageEnhance
import numpy as np


def match(src, target):    
     size = 256
     src = Image.fromarray(src)
     target = Image.fromarray(target)
     def chunk(l):
         for i in xrange(0, len(l), size):
             endpoint = i+size
             yield l[i:i+size]
         assert len(l) == endpoint # make sure chunking was complete and exhaustive
         
         
     def find_closest_idx(target, items):
         # items MUST be a monotonically increasing sequence
         last_value = None
         for i, value in enumerate(items):
             if value > target:
                 if last_value is None or abs(target - value) < abs(target - last_value):
                     return i
                 else:
                     return i - 1
             last_value = value
         return len(items)-1    
     
     src_histogram = chunk(src.histogram())
     target_histogram = chunk(target.histogram())
     
     new_levels = []
     for target_c, ref_c in zip(target_histogram, src_histogram):
         target_c = np.cumsum(target_c)
         ref_c = np.cumsum(ref_c)
         def new_level(level):
             existing_value = target_c[level]
             return find_closest_idx(existing_value, ref_c)
         new_levels.extend(map(new_level, range(0, size)))
     
     matched_target = np.asarray(target.point(new_levels))      
     
     return matched_target  

     