import sys
import os
import pathlib
import argparse
from plyfile import PlyData, PlyElement
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import numpy as np
import cv2
from tqdm import tqdm
from PointCloud import PointCloud
from Options import Options
from Editor import create_editor
def main():
    opt = Options().opt
    print(opt)
    dataset_root_path = os.path.join(opt.data_root,opt.scan)
    plyfiles_name_global = [os.path.join(opt.data_root,opt.scan,f) for f in os.listdir(dataset_root_path) if f.startswith('label_')]
    pointcloud_list = []
    for path in tqdm(plyfiles_name_global):
        pc = PointCloud()
        pc.read_from_ply(path)
        pointcloud_list.append(pc)
    merge_editor = create_editor("Merge")
    merged_pointcloud = merge_editor.merge(pointcloud_list)
    merged_pointcloud_saved_path = os.path.join(dataset_root_path,'merged.ply')
    merged_pointcloud.save_as_ply(merged_pointcloud_saved_path)
if __name__ == '__main__':
    main()
    #label:ushot

    # plydata = PlyData.read(path)
