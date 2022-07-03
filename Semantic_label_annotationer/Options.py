import sys
import os
import pathlib
import argparse
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Demo of argparse")
        parser.add_argument('--data_root',type=str, default='/home/slam/devdata/Semantic_label_annotationer/data_src',help='root of dataset')
        parser.add_argument('--scan',type=str, default='scene0113',help='Which project')
        parser.add_argument('--auto_or_manual', type=str, default='1', help='0:auto to detect blur;1:manual to detect blur')
        parser.add_argument('--num_of_remove', type=str, default='150', help='set how may blur image to be remove if use automatic')
        self.opt = parser.parse_args()
        self.opt.dataset_dir = os.path.join(self.opt.data_root,self.opt.scan,'exported')
        # print(self.opt.dataset_dir)