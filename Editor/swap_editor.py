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
        parser.add_argument('--cra',
                            type=str,
                            default='/home/vr717/Documents/qys/code/pointnerf/checkpoints/scannet/swapPoint/horse',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets a')
        parser.add_argument('--crb',
                            type=str,
                            default='/home/vr717/Documents/qys/code/pointnerf/checkpoints/scannet/swapPoint/guanZi',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets b')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2')

        self.opt = parser.parse_args()
        
        # print(self.opt.dataset_dir)

def test_load_checkpoints_save_as_ply(opt,savename):
    '''
    测试从checkpoints转ply
    '''
    cpc = CheckpointsController(opt)
    neural_pcd = cpc.load_checkpoints_as_nerualpcd()
    neural_pcd.save_as_ply(savename)
# def test_edit(opt):
#     cpc = CheckpointsController(opt)
#     scene_npcd = Neural_pointcloud(opt)
#     scene_npcd.load_from_ply('guanZi')
#     object_mpcd = Meshlab_pointcloud(opt)
#     object_mpcd.load_from_meshlabfile('sofa1')
#     object_npcd = object_mpcd.meshlabpcd2neuralpcd(scene_npcd)
#     object_npcd.save_as_ply('sofa1')
#     pce = PointCloudEditor(opt)
#     R = cauc_RotationMatrix(0, 0, 15)
#     transMatrix = cauc_transformationMatrix(R, np.array([0, -0.8, 0]))
#     transed_sofa = pce.translation_point_cloud_local(object_npcd,transMatrix)
#     transed_sofa.save_as_ply('sofa1_trans(0,0,15)(0,-0.8,0)')
#     new_scene = pce.add_point_cloud(transed_sofa,scene_npcd)
#     new_scene.save_as_ply('scene_add_sofa1(0,0,15)(0,-0.8,0)')
#     cpc.save_checkpoints_from_neuralpcd(new_scene,'scene_add_sofa1(0,0,15)(0,-0.8,0)')

# def test_edit(opt):
#     # //把horse放罐子里
#     cpc = CheckpointsController(opt)
#     scene_npcd = Neural_pointcloud(opt)
#     scene_npcd.load_from_ply('guanZi')
#     # object_mpcd = Meshlab_pointcloud(opt)
#     # object_mpcd.load_from_meshlabfile('sofa1')
#     # object_npcd = object_mpcd.meshlabpcd2neuralpcd(scene_npcd)
#     object_npcd = Neural_pointcloud(opt)
#     object_npcd.load_from_ply('horse')
#     pce = PointCloudEditor(opt)
#     R = cauc_RotationMatrix(0, 0, 0)
#     transMatrix = cauc_transformationMatrix(R, np.array([0, -1, 2]))
#     transed_sofa = pce.translation_point_cloud_local(object_npcd,transMatrix)
#     transed_sofa.save_as_ply('horse_trans(0,0,0)(0,-1,2)')
#     new_scene = pce.add_point_cloud(transed_sofa,scene_npcd)
#     new_scene.save_as_ply('scene_add_horse(0,0,0)(0,-1,2)')
#     cpc.save_checkpoints_from_neuralpcd(new_scene,'scene_add_horse(0,0,0)(0,-1,2)')

def test_edit(opt):
    # 把guanzi放马里
    cpc = CheckpointsController(opt)
    scene_npcd = Neural_pointcloud(opt)
    scene_npcd.load_from_ply('horse')
    object_mpcd = Meshlab_pointcloud(opt)
    object_mpcd.load_from_meshlabfile('guanZi')
    object_npcd = object_mpcd.meshlabpcd2neuralpcd(object_mpcd)
    object_npcd.save_as_ply('guanZi')
    pce = PointCloudEditor(opt)
    R = cauc_RotationMatrix(0, 0, 15)
    transMatrix = cauc_transformationMatrix(R, np.array([0, -0.8, 0]))
    transed_sofa = pce.translation_point_cloud_local(object_npcd,transMatrix)
    transed_sofa.save_as_ply('sofa1_trans(0,0,15)(0,-0.8,0)')
    new_scene = pce.add_point_cloud(transed_sofa,scene_npcd)
    new_scene.save_as_ply('scene_add_guanZi(0,0,15)(0,-0.8,0)')
    cpc.save_checkpoints_from_neuralpcd(new_scene,'scene_add_guanZi(0,0,15)(0,-0.8,0)')


def test_edit1(opt):# sofa only
    cpc = CheckpointsController(opt)
    object_mpcd = Meshlab_pointcloud(opt)
    scene_npcd = Neural_pointcloud(opt)
    scene_npcd.load_from_ply('scene_origin')
    object_mpcd.load_from_meshlabfile('sofa1')
    object_npcd = object_mpcd.meshlabpcd2neuralpcd(scene_npcd)
    cpc.save_checkpoints_from_neuralpcd(object_npcd,'sofa1')

def test_edit2(opt):# delete
    cpc = CheckpointsController(opt)
    object_mpcd = Meshlab_pointcloud(opt)
    scene_npcd = Neural_pointcloud(opt)
    scene_npcd.load_from_ply('scene_origin')
    object_mpcd.load_from_meshlabfile('sofa1')
    object_npcd = object_mpcd.meshlabpcd2neuralpcd(scene_npcd)
    pce = PointCloudEditor(opt)
    bg_npcd = pce.crop_point_cloud(object_npcd,scene_npcd)
    cpc.save_checkpoints_from_neuralpcd(bg_npcd,'bg_nosofa1')

def main():
    sparse = Options()
    opt = sparse.opt
    # test_load_checkpoints_save_as_ply(opt,'scene_origin')
    # 测试读ply:这一步中间，用mesh手抠一个物体，命名为sofa_meshlabpcd.ply~！~！~！~！~！~！~！~！
    # test_edit(opt)
    # test_edit1(opt)
    # test_edit2(opt)
    
    test_load_checkpoints_save_as_ply(opt,"test.ply")



if __name__=="__main__":
    main()
    print('~finish~')