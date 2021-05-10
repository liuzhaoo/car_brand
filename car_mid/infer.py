import json
import random
import os
import torchvision
import torch
import numpy as np
from torch import nn
import time
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize,ToPILImage
from pathlib import Path

from model.inceptionv4 import Inceptionv4  
from model.resnet import resnet18
from utils import get_mean_std,Logger, worker_init_fn, get_lr,AverageMeter, calculate_accuracy,calculate_union_accuracy,find_badcase
# from utils import AverageMeter, calculate_accuracy
from dataset.val_dataloader import LmdbDataset_val
from torch.utils.data._utils.collate import default_collate  # 注意版本



# def collate_fn(batch):
#     batch_clips, batch_targets,batch_keys = zip(*batch)

#     # batch_keys = [key for key in batch_keys]

#     return default_collate(batch_clips), default_collate(batch_targets), batch_keys

mid_chexing = '/home/zhaoliu/car_brand/datasets_new/maps/mid_2_chexing'
mid_label = '/home/zhaoliu/car_brand/datasets_new/maps/mid_2_label'

mid_2_chexing = {}
mid_2_label = {}
for line in open(mid_chexing,'r'):
    index,index1 = line.strip().split()
    mid_2_chexing[int(index)]=int(index1)

for line in open(mid_label,'r'):
    index,index1 = line.strip().split()
    mid_2_label[int(index1)]=index

def cal_chexing_acc(outputs, targets):
    n_correct_elems = 0

    assert len(outputs) == len(targets)
    outputs = outputs.numpy()
    outputs = np.argmax(outputs,1)

    outputs = outputs.tolist()      
    chexing_targets = targets.numpy().tolist()

    chexing_targets = [mid_2_chexing[int(target)] for target in targets]  # int
    chexing_outputs = [mid_2_chexing[int(output)] for output in outputs]  # int



    for i in range(len(outputs)):

        if str(chexing_outputs[i]) == str(chexing_targets[i]):
            n_correct_elems+=1

    return n_correct_elems/len(outputs)

def cal_ori_acc(outputs, targets):
    n_correct_elems = 0

    assert len(outputs) == len(targets)
    outputs = outputs.numpy()
    outputs = np.argmax(outputs,1)

    outputs = outputs.tolist()      
    chexing_targets = targets.numpy().tolist()

    ori_targets = []
    ori_outputs = []

    for target in chexing_targets:
        ori_t = mid_2_label[target][0]
        if ori_t == 'b':
            ori = 0
        elif ori_t == 's':
            ori = 1
        else:
            ori =2
        ori_targets.append(ori)

    for output in outputs:
        ori_o = mid_2_label[output][0]
        if ori_o == 'b':
            ori = 0
        elif ori_o == 's':
            ori = 1
        else:
            ori =2
        ori_outputs.append(ori)


    # ori_targets = [mid_2_chexing[int(target)] for target in targets]  # int
    # ori_outputs = [mid_2_chexing[int(output)] for output in outputs]  # int



    for i in range(len(outputs)):

        if str(ori_targets[i]) == str(ori_outputs[i]):
            n_correct_elems+=1

    return n_correct_elems/len(outputs)

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
                                            #  collate_fn = collate_fn,
                                             pin_memory=True)
                                             


    return test_loader



def test(model,test_loader):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    accuracies_chexing = AverageMeter()
    accuracies_ori = AverageMeter()
    end_time = time.time()


    print('----------开始测试----------')

    badcase_keys=[]
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end_time)



            # chexing = torch.tensor(targets)

            outputs= model(inputs)


            acc = calculate_accuracy(outputs, targets)
            acc_chexing = cal_chexing_acc(outputs,targets)
            acc_ori = cal_ori_acc(outputs,targets)

            accuracies_ori.update(acc_ori,inputs.size(0))
            accuracies_chexing.update(acc_chexing,inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()


            print('test iter: {}/{}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Acc_ori {acc_ori.val:.3f} ({acc_ori.avg:.3f})\t'
                      'Acc_chexing {acc_chexing.val:.3f} ({acc_chexing.avg:.3f})\t'
                      'Acc_mid {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i + 1,
                    len(test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    acc_ori = accuracies_ori,
                    acc_chexing=accuracies_chexing,
                    acc=accuracies))



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

    pth_path = '/home/zhaoliu/car_brand/car_mid/results_newdata/random_sample/save_30.pth'
    
    # test_result=Path('/home/zhaoliu/car_class+ori/test_result')
    # test_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/val_lmdb'
    # keys_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/new_val.npy'
    test_path = "/mnt/disk/zhaoliu_data/small_car_lmdb/train_val_lmdb"
    keys_path = '/home/zhaoliu/car_brand/lmdb_data_new/val.npy'


    model = resnet18(num_classes=1189)
    model = resume_model(pth_path,model)
    test_loader = get_test_utils(test_path,keys_path)
    print('数据加载完毕...')
    test(model,test_loader)

if __name__ == '__main__':
    main()
