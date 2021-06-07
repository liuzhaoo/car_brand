

import lmdb
import numpy as np
from caffe.proto import caffe_pb2



npy_path ='/home/zhaoliu/car_brand/lmdb_data_new/train.npy'
npy_path_label = '/home/zhaoliu/car_brand/datasets_new/tongji/train_tongji.npy'
lmdb_path ="/mnt/disk/zhaoliu_data/small_car_lmdb/train_val_lmdb"

txt = '/home/zhaoliu/car_brand/lmdb_data_new/weight_txt/weight.txt'

lmdb_env=lmdb.open(lmdb_path)
lmdb_txn=lmdb_env.begin()
datum = caffe_pb2.Datum()

keys_list = np.load(npy_path).tolist()
label_list= np.load(npy_path_label).tolist()


f = open(txt,'a')

# i = 0

txt_list = []

print(len(keys_list))
print(len(label_list))

# for i in range(len(keys_list)):

#     key = keys_list[i]
#     label = label_list[i]
#     item = key + '\t' + label

# for key in keys_list:
#     i +=1
#     val = lmdb_txn.get(key)
#     datum.ParseFromString(val)
#     label = datum.label

#     key = key.decode()

#     item = key + '\t' +str(label)
#     txt_list.append(item)


#     if i%1000==0:
#         print(i)

# for line in txt_list:
    # f.write(line+'\n')
f.close()
