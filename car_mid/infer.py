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

lab_gt = '/home/zhaoliu/carbrand-master/carbrand-master/dict/all_cls_2_label_dict'

lab_2_gt = {}
for line in open(lab_gt,'r'):
    index,index1 = line.strip().split()
    lab_2_gt[index1]=int(index)




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
    end_time = time.time()


    print('----------开始测试----------')

    badcase_keys=[]
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end_time)


            targets = targets.numpy().tolist()
            targets = [lab_2_gt[str(target)] for target in targets]  # int
            targets = torch.tensor(targets)

            outputs= model(inputs)


            acc = calculate_accuracy(outputs, targets)

            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()


            print('test iter: {}/{}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i + 1,
                    len(test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
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

    pth_path = '/home/zhaoliu/car_weight/car_full/results/train_weight2/save_37.pth'
    
    test_result=Path('/home/zhaoliu/car_class+ori/test_result')
    # test_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/val_lmdb'
    # keys_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/new_val.npy'
    test_path = "/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_test/allnewdata_img_test_lmdb"
    keys_path = '/home/zhaoliu/car_data/keys_npy/cleaned_test.npy'


    model = resnet18(num_classes=27249)
    model = resume_model(pth_path,model)
    test_loader = get_test_utils(test_path,test_result,keys_path)
    print('数据加载完毕...')
    test(model,test_loader)

if __name__ == '__main__':
    main()
