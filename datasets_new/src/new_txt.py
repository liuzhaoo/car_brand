"""
统计映射后的类别数，重新给定子品牌标签，同时筛选数据生成新的txt文件，使用新标签。同时保存修旧标签的映射表
"""
import random



new_txt = '/home/zhaoliu/car_brand/datasets_new/tem_files/mid_str'

new_data_path = '/home/zhaoliu/car_brand/datasets_new/tem_files/all_data.txt'   # 生成的新数据的位置
mid_map_path='/home/zhaoliu/car_brand/datasets_new/tem_files/mid_map'   # 新旧子品牌标签的映射表
with open(new_txt,'r') as f:
    lines = f.readlines()

# 统计所有类别的样本数
count = {}
for line in lines:
    item = line.strip().split('\t')
    assert len(item) == 2
    path = item[0]
    old_label = item[1]
    try:
        count[old_label].append(path)
    except:
        count[old_label] = []
        count[old_label].append(path)


new_data = open(new_data_path,'a')
mid_map = open(mid_map_path,'a')

i = 0
for k in count:

    val = count[k]   # k是旧标签 val是旧标签对应的样本路径

    if len(val) > 1000:
        random.shuffle(val)
        paths = val[:1000]
    elif len(val) >= 100:
        paths = val
    else:
        continue
    
    maps = str(i)+' '+ str(k)
    mid_map.write(maps+'\n')
    for p in paths:
        new_line = p[1:] + ' ' + str(i)  # 路径不要开头的 /  为了方便生成lmdb
        new_data.write(new_line+'\n')

    i += 1
mid_map.close()
new_data.close()




