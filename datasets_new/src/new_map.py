

new_map = '/home/zhaoliu/car_brand/datasets_new/maps/mid_2_label'
mid_map = '/home/zhaoliu/car_brand/datasets_new/tem_files/mid_map'
old_map = '/home/zhaoliu/car_old/car_data/alldata_new/all_mid_2_label_new'


new_2_old = {}
old_2_label = {}

for line in open(mid_map,'r'):
    new,old = line.strip().split()
    new_2_old[new] = old

for line in open(old_map,'r'):
    label,old = line.strip().split()
    old_2_label[old] = label

f = open(new_map,'a')
for k,v in new_2_old.items():
    item = old_2_label[v]+ ' '+ str(k)
    f.write(item+'\n')

f.close()
