

train_path = '/home/zhaoliu/car_brand/datasets/val.txt'
train_path_n = '/home/zhaoliu/car_brand/lmdb_data/val.txt'
with open(train_path,'r') as f:
    lines = f.readlines()

with open(train_path_n,'a') as f2:
    for line in lines:

        line = line.replace('\t',' ')
        line = line[1:]
        f2.write(line)
