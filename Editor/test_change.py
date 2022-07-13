import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from utils_mine.utilize import  *
#np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示

import cv2
from tqdm import tqdm
from Editor.pointcloud import *
from Editor.checkpoints_controller import *
from Editor.pointcloud_editor import *
class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Argparse of  point_editor")
        parser.add_argument('--checkpoints_root',
                            type=str,
                            default='/home/slam/devdata/NSEPN/checkpoints/scannet/61-scene0113-41+sparseview+NOgrowingPruneSemanticpointe+randomsampler(10W+10W)_finetune_edit',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2')

        self.opt = parser.parse_args()

        # print(self.opt.dataset_dir)

import copy

class ChangeOpt():
    def __init__(self, opt):
        self.opt = opt;
        print("*******************修改self*************************")
        print("内部opt修改前",self.opt.checkpoints_root)
        self.opt.checkpoints_root = "interRoot";
        print("外部opt",opt.checkpoints_root)
        print("内部opt",self.opt.checkpoints_root)
        
        print("*******************修改形参*************************")
        opt.checkpoints_root = "outRoot";
        print("外部opt",opt.checkpoints_root)
        print("内部opt",self.opt.checkpoints_root)
        
def main():
    sparse = Options()
    opt = sparse.opt

    ChangeOpt(copy.deepcopy(opt))
    print(opt.checkpoints_root)
    # test_load_checkpoints_save_as_ply(opt,'scene_origin')
    # 测试读ply:这一步中间，用mesh手抠一个物体，命名为sofa_meshlabpcd.ply~！~！~！~！~！~！~！~！



if __name__=="__main__":
    main()
    print('~finish~')