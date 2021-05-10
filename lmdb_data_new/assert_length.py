import numpy as np

txt = '/home/zhaoliu/car_brand/lmdb_data/train.txt'
npy = np.load('/home/zhaoliu/car_brand/lmdb_data/train.npy')
with open(txt,'r') as f:
    lines = f.readlines()

print(npy[0].decode())
print(lines[0])

