
from PIL import Image
from caffe.proto import caffe_pb2
import lmdb
import numpy as np
import caffe
import time
import random
from collections import defaultdict

# path = '/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/car_256_all_lmdb'


def gen_dict(keys_path):
    keys = np.load(keys_path)
    label_dict = defaultdict(list)
    for item in keys:
        path,label = item.strip().split()
        label_dict[label].append(path)

    return label_dict




def split(label_dict,train_npy,train_labels_npy,test_npy,test_labels_npy,ratio):
    #  同时保存训练用的keys和统计用的标签
    train_set=[]
    test_set=[]

    train_label_num = []
    test_label_num = []
    for label,paths in label_dict.items():
        random.shuffle(paths)
        offset = int(len(paths)*ratio)
        test_p = paths[:offset]
        train_p = paths[offset:]

        test_p_b = [test_b.encode() for test_b in test_p]
        train_p_b = [train_b.encode() for train_b in train_p]

        test_num = [label for i in range(len(test_p))]
        train_num = [label for i in range(len(train_p))]

        train_label_num.extend(train_num)
        test_label_num.extend(test_num)

        train_set.extend(train_p_b)
        test_set.extend(test_p_b)
    # train_set = 
    np.save(train_npy,train_set)
    np.save(test_npy,test_set)
    np.save(train_labels_npy,train_label_num)
    np.save(test_labels_npy,test_label_num)



if __name__ == "__main__":
# qczj_train_all_lmdb
# car_256_all_lmdb
    keys_path = '/home/zhaoliu/car_brand/lmdb_data_new/total_lmdb.npy'


    test_labels_npy = '/home/zhaoliu/car_brand/datasets_new/tongji/val_tongji.npy'
    train_labels_npy ='/home/zhaoliu/car_brand/datasets_new/tongji/train_tongji.npy'
    train_npy = '/home/zhaoliu/car_brand/lmdb_data_new/train.npy'
    test_npy = '/home/zhaoliu/car_brand/lmdb_data_new/val.npy'

    
    label_dict = gen_dict(keys_path)

    split(label_dict,train_npy,train_labels_npy,test_npy,test_labels_npy,ratio=0.1)



    # read_data(keys_path,lmdb_path)


