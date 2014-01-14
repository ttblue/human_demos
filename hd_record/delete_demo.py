import os, os.path as osp
import shutil
import re

from hd_utils.colorize import yellowprint, redprint
from hd_utils.defaults import demo_files_dir, master_name, latest_demo_name, demo_names
from hd_utils.yes_or_no import yes_or_no

"""
Deletes a demo and updates latest_demo.txt.
"""

def delete_demo(demo_type, demo_name):
    
    assert demo_type != '', 'DEMO TYPE needs to be valid.'
    assert demo_name != '', 'DEMO NAME needs to be valid.'
    
    demo_type_dir = osp.join(demo_files_dir, demo_type)
    demo_dir= osp.join(demo_type_dir, demo_name)
    
    if not osp.exists(demo_dir):
        redprint('%s does not exist! Invalid demo.'%demo_dir)
        return
    
    yellowprint('This cannot be undone!')
    if not yes_or_no('Are you still sure you want to delete %s?'%demo_name):
        return
    
    # If it was the last demo recorded, update the latest_demo.txt file
    # Otherwise don't bother
    latest_demo_file = osp.join(demo_type_dir, latest_demo_name)
    with open(latest_demo_file,'r') as fh: demo_num = int(fh.read())
    if demo_name == demo_names.base_name%demo_num:
        with open(latest_demo_file,'w') as fh: fh.write(str(demo_num-1))
    
    # Delete contents of demo_dir
    shutil.rmtree(demo_dir)
    
    # Update master_file
    master_file = osp.join(demo_type_dir, master_name)
    with open(master_file,"r") as fh: master_lines = fh.readlines()
    
    fh = open(master_file,"w")
    for line in master_lines:
      if demo_name not in line: fh.write(line)
    fh.close()
    
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type", help="Type of demonstration.", type=str)
    parser.add_argument("--demo_name", help="Name of demo to delete.", type=str)
    vals = parser.parse_args()
    
    delete_demo(vals.demo_type, vals.demo_name)
    