import os, os.path as osp
import numpy as np
import argparse
from defaults import demo_files_dir
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration", type=str)
parser.add_argument("--num_landmarks", help="Number of landmarks", type=int)


args = parser.parse_args()


demotype_dir = osp.join(demo_files_dir, args.demo_type)
input_h5file = osp.join(demotype_dir, args.demo_type+".h5")

labels_h5file = osp.join(demotype_dir, args.demo_type+"_labels.h5")
actions_h5file = osp.join(demotype_dir, args.demo_type+"_actions.h5")
test_h5file = osp.join(demotype_dir, args.demo_type+"_test.h5")
landmarks_h5file = osp.join(demotype_dir, args.demo_type+"_landmarks.h5")

input_data = h5py.File(input_h5file, 'r')
labels_data = h5py.File(labels_h5file, 'w')
actions_data = h5py.File(actions_h5file, 'w')
test_data = h5py.File(test_h5file, 'w')
landmarks_data = h5py.File(landmarks_h5file, 'w')



index = 0
for demo_name in input_data.keys():
    for seg_name in input_data[demo_name]:
        curr_demo = input_data[demo_name][seg_name]
        curr_name = demo_name+"-"+seg_name
        action_group = actions_data.create_group(curr_name)
        action_group['T_w_k'] = np.eye(4)
        
        action_group['cloud_proc_code'] = np.squeeze(curr_demo['cloud_proc_code'])
        action_group['cloud_proc_func'] = np.squeeze(curr_demo['cloud_proc_func'])
        action_group['cloud_proc_mod'] = np.squeeze(curr_demo['cloud_proc_mod'])
        action_group['cloud_xyz'] = np.squeeze(curr_demo['cloud_xyz'])
        action_group['depth'] = np.squeeze(curr_demo['depth'])
        action_group['description'] = "no description"
        action_group['leftarm'] = np.squeeze(curr_demo['l']['tfms'])
        action_group['rightarm'] = np.squeeze(curr_demo['r']['tfms'])
        action_group['l_gripper'] = np.squeeze(curr_demo['l']['pot_angles'])
        action_group['r_gripper'] = np.squeeze(curr_demo['r']['pot_angles'])
        action_group['stamps'] = np.squeeze(curr_demo['l']['stamps'])
        action_group['rgb'] = np.squeeze(curr_demo['rgb'])
        
        
        labels_group = labels_data.create_group(str(index))
        labels_group['action'] = str(curr_name)
        labels_group['cloud_xyz'] = np.squeeze(curr_demo['cloud_xyz'])
        labels_group['knot'] = str(0)
        pred_index = index if seg_name == 'seg00' else index - 1
        labels_group['pred']= str(pred_index)
        index += 1
        
        
num_states = index
landmark_ids = range(0, index)
np.random.shuffle(landmark_ids)
 

for i in range(args.num_landmarks):
    
    landmark_id = landmark_ids[i]
    landmarks_data.copy(labels_data[str(landmark_id)], str(landmark_id))
    

        

        

        
        
        
    

