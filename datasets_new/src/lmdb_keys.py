# read.py
from PIL import Image
from caffe.proto import caffe_pb2
import lmdb
import numpy as np

# path='/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/qczj_train_all_lmdb'
# path = '/mnt/data/zhaoliu_data/lmdb/carlogo_train_new/car_256_all_lmdb'
# path = "/mnt/data/zhaoliu_data/lmdb/carlogo_train_new/qczj_train_all_lmdb"
test_path = "/mnt/disk/zhaoliu_data/small_car_lmdb/youzhan_test"

def read_from_lmdb(lmdb_path, label_path):
    lmdb_env=lmdb.open(lmdb_path)
    lmdb_txn=lmdb_env.begin()
    lmdb_cursor=lmdb_txn.cursor()
    datum=caffe_pb2.Datum()
    i = 0
    keys = []
    for key,value in lmdb_cursor:
        datum.ParseFromString(value)
        i+=1
        datum.ParseFromString(value)
        label=datum.label
        # name = key.decode()

        # item = name+' '+ str(label)
        keys.append(key)
        if i % 10000 == 0:
            print('长度：%s'%i)

    keys = np.array(keys)

    np.save(label_path,keys)
    lmdb_env.close()

read_from_lmdb(test_path, '/home/zhaoliu/car_brand/lmdb_data_new/youzhan_test/youzhan_test.npy')
