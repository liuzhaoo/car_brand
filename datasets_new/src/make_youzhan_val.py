from collections import defaultdict

old_txt = '/home/zhaoliu/car_old/car_data/训练数据/4.9新加测试集/val.txt'

full_label = '/home/zhaoliu/car_old/car_data/alldata_new/all_cls_2_label_new'
# mid_label = '/home/zhaoliu/car_old/car_data/alldata_new/all_mid_2_label_new'

mid_map = '/home/zhaoliu/car_brand/datasets_new/maps/mid_2_label'

label_2_full={}
mid_2_label = {}
for lines in open(full_label,'r'):

    fullstr,label = lines.strip().split()
    label_2_full[int(label)] = fullstr

for lines in open(mid_map,'r'):
    mid,label = lines.strip().split()
    mid_2_label[mid] = int(label)


with open(old_txt) as f:
    lines = f.readlines()



label_dict = defaultdict(list)
for line in lines:

    path,label = line.strip().split()
    label_dict[label].append(path)

true = 0
false = 0
mid_txt=[]
for key,val in label_dict.items():

    

    full_str = label_2_full[int(key)]
    mid_str = '-'.join(full_str.split('-')[:2])
    try:
        mid_label = mid_2_label[mid_str]
        for path in val:
            item = path[1:] +' '+ str(mid_label)
            mid_txt.append(item)
        true+=1
    except:
        false+=1
        continue


new_val = '/home/zhaoliu/car_brand/lmdb_data_new/youzhan_test/val.txt'
with open(new_val,'a') as f:
    for i in mid_txt:
        f.write(i+'\n')

print(true)
print(false)
print(len(label_dict))



