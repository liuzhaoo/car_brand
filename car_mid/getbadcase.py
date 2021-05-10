import torch
import time
import sys
import json
import torch.distributed as dist
import numpy as np
from utils import AverageMeter, calculate_accuracy,calculate_union_accuracy,Logger

from torchvision.transforms import Compose, ToTensor, Resize, Normalize,ToPILImage
from pathlib import Path
from dataset.infer_dataloader import LmdbDataset_val
# from dataset.maps import get_maps
from model.resnet import resnet18
from torch.utils.data._utils.collate import default_collate  # 注意版本



cls_list = {}
mid_2_label ='/home/zhaoliu/car_brand/datasets/maps/mid_2_label'

for line in open(mid_2_label,'r'):
    mid,label = line.strip().split()
    cls_list[int(label)] = mid



ori ={'0':'back','1':'front','2':'side'}
chexing = {
    "0": "轿车",
    "1": "面包车",
    "2": "皮卡",
    "3": "越野车/SUV",
    "4": "商务车/MPV",
    "5": "轻型客车",
    "6": "中型客车",
    "7": "大型客车",
    "8": "公交车",
    "9": "校车",
    "10": "微型货车",
    "11": "轻型货车",
    "12": "中型货车",
    "13": "大型货车",
    "14": "重型货车",
    "15": "集装箱车",
    "16": "三轮车",
    "17": "二轮车",
    "18": "人",
    "19": "非人非车",
    "20": "叉车"}
def collate_fn(batch):
    batch_clips, batch_targets,batch_keys = zip(*batch)


    return default_collate(batch_clips), default_collate(batch_targets), batch_keys



def find_badcase(outputs, targets,keys):
    n_correct_elems = 0
    bad_keys = []
    badcase = []

    outputs = outputs.numpy()
    outputs = np.argmax(outputs,1)
    outputs = outputs.tolist()      # 车型list

    assert len(outputs) == len(targets) == len(keys)

    for i in range(len(keys)):

        if str(outputs[i]) != str(targets[i]):
            bad_keys.append(keys[i])
            bad_item = str(outputs[i]) +' '+ str(targets[i]) # 预测错误的输出和对应的真实标签
            badcase.append(bad_item)

    return bad_keys,badcase



def get_test_utils(test_path,keys_path):
    
    transform = []
    # normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
    #                                  opt.no_std_norm)
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    resize = Resize(240)
    # transform.append(ToPILImage())
    transform.append(resize)

    transform.append(ToTensor())
    transform.append(normalize)
    transform = Compose(transform)
    
    test_data = LmdbDataset_val(test_path,transform,keys_path)

    test_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=512,
                                             shuffle=False,
                                             num_workers=16, 
                                             collate_fn = collate_fn,                                 
                                             pin_memory=True)
                                             

    return test_loader

def test(model,data_loader,midnpy_path,midtxt_path):


    model.eval()


    mid_badcase_info = []
    mid_badcase_info = []
    mid_badkeys = []

    pred_cls = []
    gt_cls = []
    with torch.no_grad():
        for i, (inputs, targets_full,keys) in enumerate(data_loader):

            targets = targets_full.numpy().tolist()
            outputs_full = model(inputs)

            mid_bad_keys,mid_badcase = find_badcase(outputs_full,targets,keys)


            mid_badkeys.extend(mid_bad_keys)   # 所有车型有错误的 keys


            for j in range(len(mid_badcase)):
                info = mid_badcase[j]
                pred_mid = info.split(' ')[0]
                label_mid = info.split(' ')[1]


                pred_mid_str = cls_list[int(pred_mid)]   # 标签转换为文字
                label_mid_str = cls_list[int(label_mid)]

                item_pred = pred_mid_str+' '+pred_mid
                item_gt = label_mid_str+' '+label_mid

                pred_cls.append(item_pred)
                gt_cls.append(item_gt)

                key = mid_bad_keys[j].decode()   # 当前batch的
                item = '{}, pred_mid: {} , label_mid: {} '.format(key,pred_mid,label_mid)
                mid_badcase_info.append(item)
            
            print('{}/{}'.format(i,len(data_loader)))
            # if i == 3:
            #     break
            

        np.save(midnpy_path,mid_badkeys)
        np.save('/home/zhaoliu/car_brand/car_mid/badcase/weight_sample_s/pred_mid.npy',pred_cls)
        np.save('/home/zhaoliu/car_brand/car_mid/badcase/weight_sample_s/gt_mid.npy',gt_cls)


        with open(midtxt_path,'a') as f1:
            for item in mid_badcase_info:

                f1.write(item+'\n')



def resume_model(pth_path, model):
    print('loading checkpoint {} model'.format(pth_path))
    checkpoint = torch.load(pth_path, map_location='cpu')
    # assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def main():

    pth_path = '/home/zhaoliu/car_brand/car_mid/results/weight_sample_s/save_50.pth'
    
    test_result='/home/zhaoliu/car_brand/car_mid/badcase/weight_sample_s/'
    test_path = '/mnt/disk/zhaoliu_data/small_car_lmdb/val_lmdb'
    keys_path = '/home/zhaoliu/car_brand/lmdb_data/val.npy'


    midnpy_path = test_result + 'mid_badkeys.npy'
    midtxt_path = test_result + 'mid_badcase_info.txt'

    model = resnet18(num_classes=1507)
    model = resume_model(pth_path,model)
    test_loader = get_test_utils(test_path,keys_path)

    print('数据加载完毕...')
    test(model,test_loader,midnpy_path,midtxt_path)

if __name__ == '__main__':
    main()