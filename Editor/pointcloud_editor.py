import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np
from Editor.pointcloud import *
#np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

import cv2
from tqdm import tqdm

class PointCloudEditor:
    def __init__(self,opt,modelname=None):
        if modelname==None:
           self.opt = opt 
        else:
            opt.checkpoints_root = os.path.join(opt.checkpoints_root,modelname)
            self.opt = opt
            pass
    def crop_point_cloud(self,npcd_child,npcd_father):# remove npt2 point from npt1,return removed npt1
        '''
        Actually i dont know how to do that...
        I use every point in meshlabpcd to find the nearest pcd in neural_pcd
        '''
        # TODO: use cuda to reimplementation
        npc = Neural_pointcloud(self.opt)
        pointsize_child =npcd_child.xyz.shape[0]
        pointsize_father = npcd_father.xyz.shape[0]
        neural_xyz = np.empty([pointsize_father,3])
        neural_color = np.empty([pointsize_father,3])
        neural_embeding = np.empty([pointsize_father,32])
        neural_conf = np.empty([pointsize_father])
        neural_label = np.empty([pointsize_father])
        neural_dir = np.empty([pointsize_father,3])
        print('Scale of father neural point cloud :',(pointsize_father))
        print('Scale of child neural point cloud:',(pointsize_child))
        idx = 0
        for i in tqdm(range(pointsize_father)):
            father_ptr_xyz = npcd_father.xyz[i]
            dis = np.sqrt(np.sum(np.square(father_ptr_xyz-npcd_child.xyz),axis = -1))
            if  not (dis < 1e-7).any():
                neural_xyz[idx] = npcd_father.xyz[i]
                neural_color[idx] = npcd_father.color[i]
                neural_embeding[idx] = npcd_father.embeding[i]
                neural_conf[idx] = npcd_father.conf[i]
                neural_label[idx] = npcd_father.label[i]
                neural_dir[idx] = npcd_father.dir[i]
                idx+=1
        neural_xyz = neural_xyz[:idx]
        neural_color = neural_color[:idx]
        neural_embeding = neural_embeding[:idx]
        neural_conf = neural_conf[:idx]
        neural_dir = neural_dir[:idx]
        neural_label = neural_label[:idx]
        print('\ncrop done...neural point cloud scale:',idx)
        npc.load_from_var(neural_xyz,neural_embeding,neural_conf,neural_dir,neural_color,neural_label)
        return npc
    def translation_point_cloud_global(self,npcd, transMatirx):#rotate by world coordinate
        res_npc = Neural_pointcloud(self.opt)
        rot_matrix = transMatirx[:3, :3]
        trans_vector = transMatirx[:3, 3]
        res_npc.xyz = npcd.xyz @ rot_matrix + trans_vector
        res_npc.color = npcd.color
        res_npc.embeding = npcd.embeding
        res_npc.conf = npcd.conf
        res_npc.dir = npcd.dir@ rot_matrix
        res_npc.label = npcd.label
        return res_npc
    def translation_point_cloud_local(self,npcd,transMatirx):#rotate by self coordinate
        pointsize = npcd.xyz.shape[0]
        centerptr = np.sum(npcd.xyz,axis=0)/pointsize
        res_npc = Neural_pointcloud(self.opt)
        rot_matrix = transMatirx[:3, :3]
        trans_vector = transMatirx[:3, 3]
        res_npc.xyz = (npcd.xyz-centerptr) @ rot_matrix + trans_vector + centerptr
        res_npc.color = npcd.color
        res_npc.embeding = npcd.embeding
        res_npc.conf = npcd.conf
        res_npc.dir = npcd.dir@ rot_matrix
        # res_npc.label = npcd.label
        return res_npc
    def add_point_cloud(self,npcd_child,npcd_father):
        res_npc = Neural_pointcloud(self.opt)
        res_npc.xyz = np.concatenate((npcd_child.xyz,npcd_father.xyz),axis = 0)
        res_npc.color = np.concatenate((npcd_child.color, npcd_father.color), axis=0)
        res_npc.embeding = np.concatenate((npcd_child.embeding, npcd_father.embeding), axis=0)
        res_npc.conf = np.concatenate((npcd_child.conf, npcd_father.conf), axis=0)
        res_npc.dir = np.concatenate((npcd_child.dir, npcd_father.dir), axis=0)
        # res_npc.label = np.concatenate((npcd_child.label, npcd_father.label), axis=0)
        return res_npc