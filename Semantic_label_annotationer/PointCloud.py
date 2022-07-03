import sys
import os
import pathlib
import argparse
from plyfile import PlyData, PlyElement
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import numpy as np
import cv2
from tqdm import tqdm

class PointCloud:
    def __init__(self):
        self.label = None
        self.color = None
        self.xyz = None

    def read_from_ply(self,path):

        plydata = PlyData.read(path)
        self.color = {}
        self.color["r"], self.color["g"], self.color["b"] = np.array(plydata.elements[0].data["red"].astype(np.float32)), np.array(
            plydata.elements[0].data["green"].astype(np.float32)), np.array(
            plydata.elements[0].data["blue"].astype(np.float32))
        self.xyz={}
        self.xyz["x"],self.xyz["y"], self.xyz["z"] = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
            plydata.elements[0].data["y"].astype(np.float32)), np.array(
            plydata.elements[0].data["z"].astype(np.float32))
        self.label = np.full_like(self.xyz['x'], int(path.split('_')[-1][0]), dtype=np.uint8)
    def save_as_ply(self,path):
        assert self.label is not None, "Save empty point cloud!Check it."
        vertex = []
        print(self.label.shape)
        for i in tqdm(range(self.label.shape[0])):
            vertex.append((self.xyz["x"][i],self.xyz["y"][i],self.xyz["z"][i],self.color["r"][i],self.color["g"][i],self.color["b"][i],self.label[i]))
        vertex = np.array(vertex,dtype=[
                ("x", np.dtype("float32")),
                ("y", np.dtype("float32")),
                ("z", np.dtype("float32")),
                ("red", np.dtype("float32")),
                ("green", np.dtype("float32")),
                ("blue", np.dtype("float32")),
                ("label",np.dtype("uint8"))])
        save_pointcloud = PlyElement.describe(vertex, "vertex")
        save_pointcloud = PlyData([save_pointcloud])
        save_pointcloud.write(path)