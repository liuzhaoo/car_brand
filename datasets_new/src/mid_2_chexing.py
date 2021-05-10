


old = '/home/zhaoliu/car_old/car_data/alldata_new/mid_2_chexing'
mid_map ='/home/zhaoliu/car_brand/datasets_new/tem_files/mid_map' 
new_map = '/home/zhaoliu/car_brand/datasets_new/maps/mid_2_chexing'

old_map = {}
mid_2_mid={}
for line in open(old,'r'):
    mid,chexing = line.strip().split()
    old_map[int(mid)] = int(chexing)
for line in open(mid_map,'r'):
    new,old = line.strip().split()
    mid_2_mid[int(new)] = int(old)


with open(new_map,'a') as f:
    for k,v in mid_2_mid.items():
        chexing = old_map[k]

        item = str(k) + ' ' + str(chexing)
        f.write(item+'\n')

