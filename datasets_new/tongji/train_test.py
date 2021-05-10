import numpy as np
import matplotlib.pyplot as plt 
from collections import defaultdict
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体支持



train_txt = "/home/zhaoliu/car_brand/datasets_new/tongji/train_tongji.npy"
test_txt = '/home/zhaoliu/car_brand/datasets_new/tongji/val_tongji.npy'
# test_txt = '/home/zhaoliu/car_data/test_qczj.txt'
# with open(train_txt,'r') as f:
#     lines_train = f.readlines()

# with open(test_txt,'r') as f2:
#     lines_test = f2.readlines()
lines_train = np.load(train_txt)
lines_test = np.load(test_txt)

def sort_by_key(d):
    '''
    d.items() 返回元素为 (key, value) 的可迭代类型（Iterable），
    key 函数的参数 k 便是元素 (key, value)，所以 k[0] 取到字典的键。
    '''
    return sorted(d.items(), key=lambda k: k[1],reverse=True)

def ext_val(dict_i):
    val_ls = []
    for k,v in dict_i.items():
        val_ls.append(v)

    return val_ls

train_labels = [0]*1189
test_labels = [0]*1189

for train_label in lines_train:
    label = int(train_label.strip().split('\t')[-1])

    train_labels[label] += 1


for test_label in lines_test:
    label = int(test_label.strip().split('\t')[-1])

    test_labels[label] += 1

classes = [i for i in range(1189)]

plt.figure(figsize=(12,6), dpi=80)

ax1 = plt.subplot(211)
plt.plot(classes,train_labels, color="r",label = 'train')
plt.title('train')
plt.xlabel('classes')
plt.ylabel('num_per_class')
plt.legend()


ax1 = plt.subplot(212)
plt.plot(classes,test_labels, color="g",label = 'test')

plt.title('test')
plt.xlabel('classes')
plt.ylabel('num_per_class')
plt.legend()


plt.savefig('/home/zhaoliu/car_brand/datasets_new/tongji/train_test.jpg')


