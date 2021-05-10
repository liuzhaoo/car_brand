
import json
import numpy as np

cls_list = {}
mid_2_label ='/home/zhaoliu/car_brand/datasets_new/maps/mid_2_label'

for line in open(mid_2_label,'r'):
    mid,label = line.strip().split()
    cls_list[int(label)] = mid

def sort_by_key(d):
    '''
    d.items() 返回元素为 (key, value) 的可迭代类型（Iterable），
    key 函数的参数 k 便是元素 (key, value)，所以 k[0] 取到字典的键。
    '''
    return sorted(d.items(), key=lambda k: k[1],reverse=True)

def read_txt(file_path, save_to):
    # with open(txt_path,'r') as f:
    #     lines = f.readlines()
    lines = np.load(file_path)
    mid_dict_n = {}

    index = 0
    for item in lines:
        index += 1
        label = item.strip().split('\t')[-1]
 
        clas = cls_list[int(label)]
        # clas = cls_list[int(cls_label)]    #  根据读取到的label（27249） 得到 全品牌
        
      
        key = clas + ' '+str(label)
        try:
            mid_dict_n[key] += 1
        except:
            mid_dict_n[key] = 0
            mid_dict_n[key] += 1
        
        
        if index % 5000 ==0:
            print(index)
    full_dict = dict(sort_by_key(mid_dict_n))
    with open(save_to,'w') as js:
        json.dump(full_dict,js,indent=4,ensure_ascii=False)


if __name__ == '__main__':
    txt_path = '/home/zhaoliu/car_brand/datasets_new/tongji/val_tongji.npy'
    save_to = '/home/zhaoliu/car_brand/datasets_new/tongji/val.json'
    read_txt(txt_path, save_to)