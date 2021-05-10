
import random

def split(full_list,shuffle=True,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2


sourse_path = '/home/zhaoliu/car_brand/datasets/tem_files/all_data.txt'
train_path = '/home/zhaoliu/car_brand/datasets/train.txt'
val_path = '/home/zhaoliu/car_brand/datasets/val.txt'


with open(sourse_path,'r') as f:
    lines = f.readlines()

val,train = split(lines,ratio=0.02)

with open(train_path,'a') as f1:
    for train_item in train:
        f1.write(train_item)


with open(val_path,'a') as f2:
    for val_item in val:
        f2.write(val_item)

