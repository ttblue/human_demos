import numpy as np
import leveldb
import argparse
import os, os.path as osp
import cv2
import yaml
import shutil
import caffe
from caffe.io import caffe_pb2
from scipy import linalg
from create_leveldb_utils import *
import cPickle as cp
import h5py
from os import listdir
from shapely.geometry import Point, Polygon

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name


def sample_polygon_edge(poly):
    points = list(poly.exterior.coords)
    num_points = len(points)
    num_edges = num_points - 1
    
    edge_lengths = np.zeros(num_points)
    for i in range(num_edges):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])
        diff = p1 - p2
        edge_lengths[i+1] = np.linalg.norm(diff)
    for i in range(num_edges):
        edge_lengths[i+1] += edge_lengths[i]
        
    rand_v = np.random.uniform(0, 1) * edge_lengths[-1]
    
    for i in range(num_edges):
        if rand_v >= edge_lengths[i] and rand_v <= edge_lengths[i+1]:
            t = (rand_v - edge_lengths[i]) / (edge_lengths[i+1] - edge_lengths[i])
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            p = (1-t) * p1 + t * p2
            
            return p
        
    print "Should not be here"
    return np.array(points[-1])
        
    
        
        
    

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--num_samples", type=int)
parser.add_argument("--sample_patch_size", type=int)
parser.add_argument("--test_demo_start", type=int, default=13)
parser.add_argument("--save_image", action="store_true")
parser.add_argument("--background_accept_ratio", type=float, default=0.05)

args = parser.parse_args()
patch_size = args.sample_patch_size

task_dir = osp.join(demo_files_dir, args.demo_type)

ldbpath_train = osp.join(task_dir, "leveldb-train-rand-towel-contour-"+str(patch_size))
ldbpath_test = osp.join(task_dir, "leveldb-test-rand-towel-contour-"+str(patch_size))

if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)

# save sampled images for test
if args.save_image:
    imagepath_train = osp.join(task_dir, "train-rand-towel-contour-"+str(patch_size))
    imagepath_test = osp.join(task_dir, "test-rand-towel-contour-"+str(patch_size))

    if osp.exists(imagepath_train):
        shutil.rmtree(imagepath_train)
    os.mkdir(imagepath_train)
    os.mkdir(osp.join(imagepath_train, "background"))
    os.mkdir(osp.join(imagepath_train, "face"))
    os.mkdir(osp.join(imagepath_train, "edge"))

    if osp.exists(imagepath_test):
        shutil.rmtree(imagepath_test)
    os.mkdir(imagepath_test)
    os.mkdir(osp.join(imagepath_test, "background"))
    os.mkdir(osp.join(imagepath_test, "face"))
    os.mkdir(osp.join(imagepath_test, "edge"))

imagepath = osp.join(task_dir, "towel-image")
image_indices = listdir(imagepath)
image_indices.sort()

print image_indices

num_images = len(image_indices)
batch_train = leveldb.WriteBatch()
batch_test = leveldb.WriteBatch()

print num_images



# in the order of
# ['IMG_1016.JPG', 'IMG_1017.JPG', 'IMG_1018.JPG', 'IMG_1019.JPG', 
# 'IMG_1020.JPG', 'IMG_1022.JPG', 'IMG_1023.JPG', 'IMG_1024.JPG', 
# 'IMG_1025.JPG', 'IMG_1026.JPG', 'IMG_1027.JPG', 'IMG_1028.JPG', 'IMG_1029.JPG']



image_polygons_hw_reverted = [[(1017, 527), (2532, 620), (4338, 746), (4673, 1814), (4960, 2606), (3921, 2684), (2763, 2713), (2133, 2700), (597, 2623),(747, 1760)],
                              [(1378, 779), (2169, 855), (2235, 826), (2228, 798), (2298, 706), (2431, 598), (2637, 674), (2999, 539), (3339, 610), (3485, 599), (3622, 1399), (3772, 1468), (3775, 1539), (3744, 1924), (3766, 2499),
                               (3254, 2231), (3102, 2160), (3027, 2505), (2870, 2744), (2713, 2627), (2423, 2709), (2048, 2524), (1738, 2457), (1532, 2456), (1279, 2536), (1244, 2453), (1470, 2146), (1251, 1706), (1229, 1577),
                               (1186, 1418)],
                              [(640, 544), (2054, 574), (2197, 734), (2694, 687), (3068, 683), (3769, 595), (4236, 1756), (4734, 2784), (4602, 2817), (4137, 2527), (3755, 2611), (2809, 2592), (2636, 2595), (1391, 2565), (443, 2568),
                               (584, 1619), (606, 1467), (654, 1190), (611, 731)],
                              [(1047, 560), (1645, 590), (1955, 644), (2138, 456), (2333, 443), (3322, 510), (3972, 475), (4516, 602), (4579, 1524), (4666, 2067), (4752, 2776), (2618, 2098), (1376, 2415), (1009, 2437), (407, 2793),
                               (702, 1565), (997, 689)],
                              [(550, 995), (2288, 945), (2993, 962), (4440, 1022), (4559, 1536), (4570, 1598), (4613, 1601), (4848, 2192), (4836, 2362), (2480, 2517), (617, 2538), (202, 2511), (252, 2138), (398, 1606), (476, 1367)],
                              [(453, 855), (605, 872), (1289, 651), (2299, 189), (2915, 869), (4244, 858), (4418, 1494), (4586, 2008), (5005, 2979), (3777, 2571), (2552, 2040), (745, 1197), (404, 971)],
                              [(825, 526), (2312, 536), (3228, 587), (4443, 667), (4582, 1286), (4106, 1898), (3287, 2779), (3093, 2860), (2286, 2853), (1575, 2821), (1294, 2715), (794, 1525), (684, 1135)],
                              [(1050, 1503), (2451, 435), (3626, 671), (4322, 1541), (4446, 1701), (4571, 2208), (4656, 2975), (2831, 2688), (2549, 2632), (1444, 2428), (659, 2296), (212, 2374), (218, 2236)],
                              [(1580, 551), (1871, 565), (1991, 655), (2257, 667), (2354, 586), (2372, 406), (2607, 378), (2908, 590), (2951, 629), (3099, 670), (3094, 692), (3388, 547), (3523, 643), (3777, 652), (3843, 1202), (3789, 1242),
                               (3837, 1524), (3910, 1768), (4073, 2175), (4278, 2646), (3855, 2779), (3406, 2808), (3153, 2604), (2859, 2770), (2515, 2824), (2373, 2501), (2292, 2300), (2202, 2492), (1993, 2507), (1711, 2454),
                               (1598, 2698), (1111, 2744), (1128, 2276), (1324, 1709), (1386, 1491), (1401, 1374), (1388, 1132)],
                              [(1020, 547), (1266, 525), (1454, 580), (2031, 539), (2327, 452), (2786, 592), (2991, 705), (3173, 678), (3251, 725), (3604, 597), (3989, 528), (4182, 1274), (4246, 1528), (4400, 2077), (4568, 2818),
                               (3690, 2786), (3511, 2606), (3226, 2682), (2839, 2851), (2513, 2868), (2225, 2571), (2229, 2448), (2145, 2343), (2070, 2526), (1549, 2740), (859, 2945), (689, 2964), (651, 2455), (749, 1699),
                               (837, 1495), (903, 1203)],
                              [(781, 650), (964, 632), (1933, 623), (2955, 679), (3233, 623), (3476, 703), (4293, 636), (4475, 1473), (4620, 2051), (4842, 2800), (3785, 2845), (3070, 2914), (2491, 2895), (2071, 2749), (1813, 2772),
                               (1244, 2864), (598, 2952), (361, 2970), (493, 2498), (586, 1899), (655, 1541), (731, 986)],
                              [(784, 693), (2804, 714), (4232, 688), (4454, 1541), (4627, 2142), (4807, 2700), (3490, 2845), (3252, 2867), (2511, 2902), (1016, 2903), (697, 2906), (407, 2894), (417, 2745), (538, 2044), (645, 1547),
                               (709, 1166)],
                              [(728, 597), (2105, 594), (3152, 565), (3412, 627), (4341, 658), (4415, 1478), (4475, 1944), (4620, 2781), (2992, 2700), (2667, 2543), (2618, 2386), (2368, 2562), (1460, 2903), (372, 2939), (505, 1929), (658, 1123)]]

image_polygons = []
for i in range(len(image_polygons_hw_reverted)):
    poly_reverted = image_polygons_hw_reverted[i]
    poly = []
    for j in range(len(poly_reverted)):
        point = poly_reverted[j]
        poly.append((point[1], point[0]))
    
    image_polygons.append(poly)
    


polygons = []
for i in range(len(image_polygons)):
    polygons.append(Polygon(image_polygons[i]))
    


#for i in range(args.num_samples):
i = 0
while i < args.num_samples:
    image_id = i % num_images
    demo_name = image_indices[image_id]
        
    image = cv2.imread(osp.join(imagepath, demo_name))
    
    (h, w, c) = image.shape
    print h, w, c
    
    
    poly = polygons[image_id]
    
    rnd_v = np.random.uniform(0, 1)
    if rnd_v < 0.5:
        # sample edge/corner points
        center_point = sample_polygon_edge(poly)
        center_point[0] += np.random.randint(-patch_size / 2, patch_size / 2)
        center_point[1] += np.random.randint(-patch_size / 2, patch_size / 2)
        start_x = int(center_point[0] - patch_size / 2.0)
        start_y = int(center_point[1] - patch_size / 2.0)
        class_name = 'edge'
    else: 
        # sample background / face points
        if rnd_v > 0.9: 
            # sample background
            while True:
                start_x = np.random.randint(0, h - patch_size)
                start_y = np.random.randint(0, w - patch_size)
                center_x = start_x + int(patch_size / 2.0)
                center_y = start_y + int(patch_size / 2.0)
                
                if poly.contains(Point(center_x, center_y)):
                    continue
                else:
                    break
            class_name = 'background'
        else:
            # sample face
            while True:
                start_x = np.random.randint(0, h - patch_size)
                start_y = np.random.randint(0, w - patch_size)
                center_x = start_x + int(patch_size / 2.0)
                center_y = start_y + int(patch_size / 2.0)
                
                if poly.contains(Point(center_x, center_y)):
                    break
                else:
                    continue
            class_name = 'face'
            
    if start_x < 0 or start_x >= h - patch_size or start_y < 0 or start_y >= w - patch_size:
        continue
            
    patch_img = image[start_x:start_x + patch_size, start_y:start_y + patch_size, :]
    
    
    patch = cv2datum(patch_img)
    patch = caffe.io.array_to_datum(patch)
    patch = patch.SerializeToString()
    
    
    patch_key = str(i)
    
    if image_id < args.test_demo_start:
        batch_train.Put(patch_key, patch)
        #cv2.imwrite(osp.join(imagepath_train, patch_key+".jpg"), patch_img)
    else:
        batch_test.Put(patch_key, patch)
        #cv2.imwrite(osp.join(imagepath_test, patch_key+".jpg"), patch_img)
        
    if args.save_image:
        if image_id < args.test_demo_start:
            cv2.imwrite(osp.join(imagepath_train, class_name, patch_key+".jpg"), patch_img)
        else:
            cv2.imwrite(osp.join(imagepath_test, class_name, patch_key+".jpg"), patch_img)
            
    i = i + 1
        
ldb_train.Write(batch_train, sync=True)
ldb_test.Write(batch_test, sync=True)