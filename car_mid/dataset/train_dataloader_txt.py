import caffe
import torch
import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from caffe.proto import caffe_pb2
import lmdb
import accimage
# from proto import tensor_pb2
# from proto import utils

class TxtDataset_train(Dataset):
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

if __name__ == '__main__':
    
    data = TxtDataset_train('/mnt/disk/zhaoliu_data/carlogo/train_data/carlogo_train.txt',None)

    print(len(data))
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=16,
                                               shuffle=False,
                                               num_workers=8,
                                               pin_memory=True)

    print(len(train_loader))