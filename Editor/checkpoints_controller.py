import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np
#np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from Editor.pointcloud import  *
import cv2
from tqdm import tqdm

class CheckpointsController:
    def __init__(self,opt,modelname = None):
        if modelname==None:
            self.opt = opt
            self.edit_dir = os.path.join(opt.checkpoints_root,'edit')
            if 'edit' not in os.listdir(opt.checkpoints_root):
                os.makedirs(os.path.join(self.edit_dir))
            self.checkpoints_root = opt.checkpoints_root
            self.device = 'cuda:'+opt.gpu_ids
            pth_files_dir = [f for f in os.listdir(self.checkpoints_root) if f.endswith('_net_ray_marching.pth')]
            assert len(pth_files_dir)>0,'Empty checkpoints file'
            numlist = [int(i.split('_')[0]) for i in pth_files_dir]
            numlist.sort()
            self.latest_iters = str(numlist[-1])
            self.latest_file_path = os.path.join(self.checkpoints_root,self.latest_iters+'_net_ray_marching.pth')# find the latest pth file
        else:
            # 重新生成opt文件 防止修改外部
            
            opt.checkpoints_root = os.path.join(opt.checkpoints_root,modelname)
            print(opt.checkpoints_root)
            self.opt = opt
            self.edit_dir = os.path.join(opt.checkpoints_root,'edit')
            print(opt.checkpoints_root)
            if 'edit' not in os.listdir(opt.checkpoints_root):
                os.makedirs(os.path.join(self.edit_dir))
            self.checkpoints_root = opt.checkpoints_root
            self.device = 'cuda:'+opt.gpu_ids
            pth_files_dir = [f for f in os.listdir(self.checkpoints_root) if f.endswith('_net_ray_marching.pth')]
            assert len(pth_files_dir)>0,'Empty checkpoints file'
            numlist = [int(i.split('_')[0]) for i in pth_files_dir]
            numlist.sort()
            self.latest_iters = str(numlist[-1])
            self.latest_file_path = os.path.join(self.checkpoints_root,self.latest_iters+'_net_ray_marching.pth')# find the latest pth file

    # find the latest pth file in root of dataset and read that file
    def load_checkpoints_as_nerualpcd(self):
        print(self.latest_file_path)
        network_paras = torch.load(self.latest_file_path, map_location=torch.device('cpu'))
        print('loading checkpoints...',self.latest_file_path)
        self.points_xyz = network_paras["neural_points.xyz"].view(-1,3).cpu().numpy()
        self.points_embeding = network_paras["neural_points.points_embeding"].view(-1,32).cpu().numpy()
        self.points_conf= network_paras["neural_points.points_conf"].view(-1,1).cpu().numpy()
        self.points_dir = network_paras["neural_points.points_dir"].view(-1,3).cpu().numpy()
        self.points_color = network_paras["neural_points.points_color"].view(-1,3).cpu().numpy()
        # self.points_label = network_paras["neural_points.points_label"].view(-1, 1).cpu().numpy()
        print('point cloud scale:',self.points_color.shape,type(self.points_color))
        neural_pcd = Neural_pointcloud(opt=self.opt)
        neural_pcd.load_from_checkpoints(self.points_xyz,self.points_embeding,self.points_conf,self.points_dir,self.points_color)
        return neural_pcd
    
    # def load_checkpoints_as_nerualpcd(self,opt,model_name):
    #     opt.checkpoints_root = os.path.join( opt.checkpoints_root,model_name)
    #     print("self重新init前的checkpoints rout:",self.checkpoints_root)
    #     self.__init__(opt);
    #     print("self重新init后的checkpoints rout:",self.checkpoints_root)
    #     network_paras = torch.load(self.latest_file_path, map_location=torch.device('cpu'))
    #     print('loading checkpoints...',self.latest_file_path)
    #     self.points_xyz = network_paras["neural_points.xyz"].view(-1,3).cpu().numpy()
    #     self.points_embeding = network_paras["neural_points.points_embeding"].view(-1,32).cpu().numpy()
    #     self.points_conf= network_paras["neural_points.points_conf"].view(-1,1).cpu().numpy()
    #     self.points_dir = network_paras["neural_points.points_dir"].view(-1,3).cpu().numpy()
    #     self.points_color = network_paras["neural_points.points_color"].view(-1,3).cpu().numpy()
    #     # self.points_label = network_paras["neural_points.points_label"].view(-1, 1).cpu().numpy()
    #     print('point cloud scale:',self.points_color.shape,type(self.points_color))
    #     neural_pcd = Neural_pointcloud(self.opt)
    #     neural_pcd.load_from_checkpoints(self.points_xyz,self.points_embeding,self.points_conf,self.points_dir,self.points_color)
    #     return neural_pcd
    
        
    def save_checkpoints_from_neuralpcd(self,neural_pcd,name):
        print('Saving checkpoints from neural point cloud...')
        network_paras = torch.load(self.latest_file_path, map_location=self.device)
        network_paras["neural_points.xyz"] = torch.Tensor(neural_pcd.xyz)  #[ptr,3]
        network_paras["neural_points.points_embeding"] = torch.unsqueeze(torch.Tensor(neural_pcd.embeding),dim=0) #[1,ptr,32]
        network_paras["neural_points.points_conf"] =  torch.unsqueeze(torch.Tensor(neural_pcd.conf[...,np.newaxis]),dim=0)#[1,ptr,1]
        network_paras["neural_points.points_dir"] = torch.unsqueeze(torch.Tensor(neural_pcd.dir),dim=0)#[1,ptr,3]
        network_paras["neural_points.points_color"] = torch.unsqueeze(torch.Tensor(neural_pcd.color),dim=0) #[1,ptr,3]
        # network_paras["neural_points.points_label"] = torch.Tensor(neural_pcd.label[...,np.newaxis])  # [ptr,3]
        torch.save(network_paras,os.path.join(self.checkpoints_root,'edit',self.latest_iters+'_net_ray_marching_Edited_'+name+'.pth'))# find the latest pth file)
        print('Saving checkpoints done')

    def save_checkpoints_from_neuralpcd_cover_by_b(self,modelB_file_path):
        print('Saving checkpoints from neural point cloud Cover by %s',modelB_file_path)
        network_paras = torch.load(self.latest_file_path, map_location=self.device)
        b_paras = torch.load(modelB_file_path, map_location=self.device)
        network_paras["aggregator.block1.0.weight"] = b_paras["aggregator.block1.0.weight"]
        network_paras["aggregator.block1.0.bias"] =  b_paras["aggregator.block1.0.bias"]
        network_paras["aggregator.block1.2.weight"] = b_paras["aggregator.block1.2.weight"] 
        network_paras["aggregator.block1.2.bias"] = b_paras["aggregator.block1.2.bias"]
        network_paras["aggregator.block3.0.weight"] = b_paras["aggregator.block3.0.weight"]
        network_paras["aggregator.block3.0.bias"] = b_paras["aggregator.block3.0.bias"]
        
        network_paras["aggregator.block3.2.weight"]  = b_paras["aggregator.block3.2.weight"] 
        network_paras["aggregator.block3.2.bias"] = b_paras["aggregator.block3.2.bias"]
        
        network_paras["aggregator.alpha_branch.0.weight"]  = b_paras["aggregator.alpha_branch.0.weight"] 
        network_paras["aggregator.alpha_branch.0.bias"] = b_paras["aggregator.alpha_branch.0.bias"]
        
        network_paras["aggregator.color_branch.0.weight"]  = b_paras["aggregator.color_branch.0.weight"] 
        network_paras["aggregator.color_branch.0.bias"] = b_paras["aggregator.color_branch.0.bias"]
        
        network_paras["aggregator.color_branch.2.weight"] = b_paras["aggregator.color_branch.2.weight"] 
        network_paras["aggregator.color_branch.2.bias"] = b_paras["aggregator.color_branch.2.bias"]
        
        network_paras["aggregator.color_branch.4.weight"] = b_paras["aggregator.color_branch.4.weight"]
        network_paras["aggregator.color_branch.4.bias"] = b_paras["aggregator.color_branch.4.bias"]
        
        network_paras["aggregator.color_branch.6.weight"]  = b_paras["aggregator.color_branch.6.weight"]
        network_paras["aggregator.color_branch.6.bias"]  = b_paras["aggregator.color_branch.6.bias"]
        
        try:
            os.rename(self.latest_file_path,self.latest_file_path+"_bk")
        except:
            print("error,"+self.latest_file_path+"_bk")
        torch.save(network_paras,os.path.join(self.checkpoints_root,'edit',self.latest_iters+'_net_ray_marching_Edited_.pth'))# find the latest pth file)
        torch.save(network_paras,os.path.join(self.checkpoints_root,self.latest_iters+'_net_ray_marching.pth'))# find the latest pth file)
        print('Saving checkpoints done')