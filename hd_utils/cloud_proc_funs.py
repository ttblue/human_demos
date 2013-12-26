import clouds
import cv2
from defaults import asus_xtion_pro_f as f
# import skimage.morphology as skim

DEBUG_PLOTS=False

def extract_red(rgb, depth):
    """
    extract red points and downsample
    """
        
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    h_mask = (h<15) | (h>145)
    s_mask = (s > 30 )
    v_mask = (v > 100)
    red_mask = h_mask & s_mask & v_mask
    
    valid_mask = depth > 0
    
    xyz = clouds.depth_to_xyz(depth, f)
    
    good_mask = red_mask & valid_mask
    # good_mask = skim.remove_small_objects(good_mask, min_size=64)

    if DEBUG_PLOTS:
        cv2.imshow("hue", h_mask.astype('uint8')*255)
        cv2.imshow("sat", s_mask.astype('uint8')*255)
        cv2.imshow("val", v_mask.astype('uint8')*255)
        cv2.imshow("final",good_mask.astype('uint8')*255)
        cv2.imshow("rgb", rgb)
        cv2.waitKey()
            
    good_xyz = xyz[good_mask]
    

    return clouds.downsample(good_xyz, .025)