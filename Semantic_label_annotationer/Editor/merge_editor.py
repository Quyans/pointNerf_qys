from .base_editor import BaseEditor
from PointCloud import PointCloud
import numpy as np
class MergeEditor(BaseEditor):
    def __init__(self):
        super().__init__()
    def merge(self,pointcloud_list) -> PointCloud:
        merged_point_cloud = PointCloud()
        for pc in pointcloud_list:
            if merged_point_cloud.label is None:# the first pointcloud
                merged_point_cloud = pc
            else:
                merged_point_cloud.label = np.concatenate([merged_point_cloud.label,pc.label],axis = 0)
                merged_point_cloud.xyz['x'] = np.concatenate([merged_point_cloud.xyz['x'], pc.xyz['x']], axis=0)
                merged_point_cloud.xyz['y'] = np.concatenate([merged_point_cloud.xyz['y'], pc.xyz['y']], axis=0)
                merged_point_cloud.xyz['z'] = np.concatenate([merged_point_cloud.xyz['z'], pc.xyz['z']], axis=0)
                merged_point_cloud.color['r'] = np.concatenate([merged_point_cloud.color['r'], pc.color['r']], axis=0)
                merged_point_cloud.color['g'] = np.concatenate([merged_point_cloud.color['g'], pc.color['g']], axis=0)
                merged_point_cloud.color['b'] = np.concatenate([merged_point_cloud.color['b'], pc.color['b']], axis=0)
        return merged_point_cloud