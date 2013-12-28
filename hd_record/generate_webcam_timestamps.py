import os, os.path as osp

def gen_timestamps (video_dir):
    init_ts = None
    with open(osp.join(video_dir,'stamps_init.txt'),'r') as fh:
        init_timestamp = float(fh.read())
    
    with open(osp.join(video_dir,'stamps_info.txt'),'r') as fh:
        ts_data = fh.readlines()
        
    stamps_file = open(osp.join(video_dir,'stamps.txt'),'w')
    
    for line in ts_data:
        if 'filename' in line:
            idx = line.find('timestamp')
            idx2 = line.find(')',idx)
            idx3 = line.find(',',idx2)
            ts_now = float(line[idx2+1:idx3])/1.0e9
            stamps_file.write(str(init_timestamp+ts_now)+'\n')
    
    stamps_file.close()
            