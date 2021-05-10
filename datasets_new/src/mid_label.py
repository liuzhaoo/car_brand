



"""
解析出油站数据的列表，按照路径和原标签顺序，然后根据原标签映射到子品牌，保存到txt文件。
"""

full_label_path = '/home/zhaoliu/car_old/car_data/alldata_new/all_cls_2_label_new'
mid_2_label_path = '/home/zhaoliu/car_old/car_data/alldata_new/all_mid_2_label_new'

# json_file = '/home/zhaoliu/car_brand/datasets/all_train.json'

full_2_label = {}
mid_dict={}
for line in open(full_label_path,'r'):
    full_str,idx = line.strip().split(' ',1)
    full_2_label[int(idx)] = full_str
for line in open(mid_2_label_path,'r'):
	mid_cls, idx = line.strip().rsplit(' ', 1)
	mid_dict[mid_cls] = int(idx)  # 子品牌-label(9186) 


train_all_path = "/mnt/disk/zhaoliu_data/carlogo/train_data/carlogo_train.txt"
train_qczj_path = "/mnt/disk/zhaoliu_data/carlogo/carlogo_train.txt"
mid_str_path = '/home/zhaoliu/car_brand/datasets/tem_files/mid_str'

with open(train_all_path,'r') as f1:
    lines_all = f1.readlines()
with open(train_qczj_path,'r') as f2:
    lines_qczj = f2.readlines()

# print(len(lines_all)) 
# print(len(lines_qczj))

lines_all = lines_all[len(lines_qczj):]

with open(mid_str_path,'a') as f3:
    for item in lines_all:
        item = item.strip().split(' ')
        path = item[0]
        label = item[1]

        try:
            full = full_2_label[int(label)]

            x = '-'.join(full.split('-')[:2])
            mid_label = mid_dict[x]
        except:
            continue

        new_line = path+'\t'+str(mid_label)

        f3.write(new_line+'\n')

