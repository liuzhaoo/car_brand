import json
import matplotlib.pyplot as pyplot
import numpy as np
from collections import defaultdict
all_train_set ="/mnt/disk/zhaoliu_data/carlogo/carlogo_data/merge_delete_train_list/new_train_set_all.lst"

full_label_path = '/home/zhaoliu/car_old/car_data/alldata_new/all_cls_2_label_new'
mid_2_label_path = '/home/zhaoliu/car_old/car_data/alldata_new/all_mid_2_label_new'

json_file = '/home/zhaoliu/car_brand/datasets/all_train.json'

full_2_label = {}
mid_dict={}

for line in open(full_label_path,'r'):
    full_str,idx = line.strip().split(' ',1)
    full_2_label[int(idx)] = full_str
for line in open(mid_2_label_path,'r'):
	mid_cls, idx = line.strip().rsplit(' ', 1)
	mid_dict[mid_cls] = int(idx)  # 子品牌-label(9186) 
def sort_by_key(d):
    '''
    d.items() 返回元素为 (key, value) 的可迭代类型（Iterable），
    key 函数的参数 k 便是元素 (key, value)，所以 k[0] 取到字典的键。
    '''
    return sorted(d.items(), key=lambda k: k[1],reverse=True)


with open(all_train_set,'r') as f:
    lines = f.readlines()


train_clsass = defaultdict(lambda:0)


for line in lines:
    full_label = int(line.strip().split(' ')[-1])
    full = full_2_label[full_label]
    mid_str = '-'.join(full.split('-')[:2])
    try:
        mid = mid_dict[mid_str]
    except:
        print(full)
    key = mid_str + ' '+ str(mid)
    
    train_clsass[key]+=1

train_clsass = dict(sort_by_key(train_clsass))
with open(json_file,'w') as f1:
    json.dump(train_clsass,f1,indent =4,ensure_ascii=False)



    




