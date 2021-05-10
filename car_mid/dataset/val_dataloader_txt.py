import caffe
import torch

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from caffe.proto import caffe_pb2

import lmdb

class TxtDataset_val(Dataset):
    def __init__(self,txt_path,optimizer):
        # super().__init__()
        self.optimizer = optimizer
        self.img_path,self.labels = self.__make_dataset(txt_path)
    def __make_dataset(self,txt_path):
        with open(txt_path,'r') as f:
            lines = f.readlines()
        path_list = []
        label_list = []

        for line in lines:

            line = line.strip().split('\t')
            try:
                path = line[0]
                label = int(line[1])    # 过滤文件名中含空格的样本
            except:
                continue
            path_list.append(path)
            label_list.append(label)
        
        return path_list,label_list


        
    def __getitem__(self, index):
        image=Image.open(self.img_path[index])

        image =self.optimizer(image)

        label= self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


