import caffe
import torch
import pyarrow as pa
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from caffe.proto import caffe_pb2
# from proto import tensor_pb2
# from proto import utils
import lmdb


class LmdbDataset_val(Dataset):
    def __init__(self,lmdb_path,optimizer,keys_path):
        # super().__init__()

        self.optimizer = optimizer
        self.datum=caffe_pb2.Datum()
        self.lmdb_path = lmdb_path
        keys = np.load(keys_path)
        self.keys = keys.tolist()
        self.length = len(self.keys)
        
    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True,write=False)
   

            

        
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        serialized_str = self.txn.get(self.keys[index])
        self.datum.ParseFromString(serialized_str)

        
        size=self.datum.width*self.datum.height

        pixles1=self.datum.data[0:size]
        pixles2=self.datum.data[size:2*size]
        pixles3=self.datum.data[2*size:3*size]

        image1=Image.frombytes('L', (self.datum.width, self.datum.height), pixles1)
        image2=Image.frombytes('L', (self.datum.width, self.datum.height), pixles2)
        image3=Image.frombytes('L', (self.datum.width, self.datum.height), pixles3)

        img=Image.merge("RGB",(image3,image2,image1))
        # img = caffe.io.datum_to_array(datum)
        
        # img=Image.fromarray(img)
        img =self.optimizer(img)

        label=self.datum.label
        return img, label,self.keys[index]

    def __len__(self):
        return self.length

# keys_path = '/home/zhaoliu/car_class/dataset/qczj_train_all_lmdb.npy'
# dataset1 = LmdbDataset_val("/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/qczj_train_all_lmdb",keys_path)
# # dataset2 = LmdbDataset("/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/qczj_train_all_lmdb")
# # dataset2 = LmdbDataset("/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/car_256_all_lmdb")

# # _DATASETS = {
# #     'd1': dataset1,
# #     'd2': dataset2,    
# # }
# # dataset = dataset1+dataset2
# dataloader = DataLoader(dataset1, batch_size=256, shuffle=True,
#                                      num_workers=16)


# for i, data in enumerate(dataloader):
#     img, label = data
#     print(i, img.shape, label.shape)

