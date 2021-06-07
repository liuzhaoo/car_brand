import json
import random
import os
import time
from re import I, T, split
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler, Adam
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Resize, Normalize,RandomRotation,RandomCrop,RandomHorizontalFlip
from pathlib import Path
 
from model.racnn import RACNN
from dataloader import LmdbDataset
from opts import parse_opts
from utils import Logger, get_lr,save_img

from train_apn import apn_train_epoch
from train_cls import cls_train_epoch
from validation import val_epoch

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def save_checkpoint(save_file_path, epoch, model, optimizer_cls,optimizer_apn, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer_cls': optimizer_cls.state_dict(),
        'optimizer_apn': optimizer_apn.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def get_opt():
    opt = parse_opts()
    opt.begin_epoch = 1
    opt.n_input_channels = 3
    opt.

    print(opt)
    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt

def get_train_utils(opt, cls_params,apn_params):

    training_data = LmdbDataset(opt.train_data,opt.keys_path_train,'train')

    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               pin_memory=True)

    train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc_cls', 'acc_apn','lr'])

    if opt.optimizer == 'sgd':
        optimizer_cls = SGD(cls_params,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)
        optimizer_apn = SGD(apn_params,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer_cls = Adam(cls_params,
                        lr=opt.learning_rate)
        optimizer_apn = Adam(apn_params,
                        lr=opt.learning_rate)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer_cls, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer_cls,
                                             opt.multistep_milestones)

    return (train_loader, train_logger, optimizer_cls,optimizer_apn, scheduler)

def get_val_utils(opt):
    
    val_data = LmdbDataset(opt.val_data,opt.keys_path_val,'test')

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_threads,
                                             pin_memory=True)



    val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc_cls','acc_apn'])

    return val_loader, val_logger

def get_weak_loc(features):
    ret = []   # search regions with the highest response value in conv5
    for i in range(len(features)):
        # resize = 224 if i >= 1 else 448
        resize = 224 

        response_map_batch = F.interpolate(features[i], size=[resize, resize], mode="bilinear").mean(1)  # mean alone channels
        ret_batch = []
        for response_map in response_map_batch:
            argmax_idx = response_map.argmax()
            ty = (argmax_idx % resize)
            argmax_idx = (argmax_idx - ty)/resize
            tx = (argmax_idx % resize)
            ret_batch.append([(tx*1.0/resize).clamp(min=0.25, max=0.75), (ty*1.0/resize).clamp(min=0.25, max=0.75), 0.25])  # tl = 0.25, fixed
        ret.append(torch.Tensor(ret_batch))
    return ret


def pretrainAPN(trainloader,optimizer_apn,opt,model,tb_writer):
    for epoch in range(2):
        for step, (inputs, _) in enumerate(trainloader, 0):
            t0 = time.time()
            inputs = Variable(inputs).to(opt.device)
            _, features, attens, _ = model(inputs)
            weak_loc = get_weak_loc(features)
            optimizer_apn.zero_grad()
            loss = F.smooth_l1_loss(attens[0], weak_loc[0].to(opt.device))


            loss.backward()
            optimizer_apn.step()
            iteration = epoch * int(len(trainloader)) + (step + 1)

            t1 = time.time()
            
            # if (step % 20) == 0:
            print(" [*] pre_apn_epoch[%d], || pre_apn_iter %d || pre_apn_loss: %.4f || Timer: %.4fsec"%(epoch, iteration, loss.item(), (t1 - t0)))

            tb_writer.add_scalar('pre_apn_loss', loss.item(), iteration)

            if loss.item() <=0.00009:
                break


def main_worker(opt):
    
    # opt.device = torch.device('cuda')

    model = RACNN(num_classes=opt.num_classes)
    model.to(opt.device)
    # model = torch.nn.DataParallel(model,device_ids=[0,1])
    print(model)
    cls_params = list(model.b1.parameters()) + list(model.b2.parameters()) + list(model.classifier1.parameters()) + list(model.classifier2.parameters())
    apn_params = list(model.apn.parameters())
    # optimizer = model.parameters()
    criterion = CrossEntropyLoss().to(opt.device)

    (train_loader, train_logger, optimizer_cls,optimizer_apn, scheduler) = get_train_utils(opt, cls_params,apn_params)
    val_loader, val_logger = get_val_utils(opt)


    test_sample, _ = next(iter(val_loader))

    tb_writer = SummaryWriter(log_dir=opt.result_path)
    # pretrainAPN(train_loader,optimizer_apn,opt,model,tb_writer)
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
   
        cls_train_epoch(i, train_loader, model, criterion, optimizer_cls,
                    opt.device, train_logger,tb_writer)
        apn_train_epoch(i, train_loader, model, optimizer_apn,
                    opt.device,tb_writer)

        if i % opt.checkpoint == 0:
            save_file_path = opt.result_path / 'save_{}.pth'.format(i)
            save_checkpoint(save_file_path, i,model, optimizer_cls,optimizer_apn,
                            scheduler)

        # if i % 5 == 0:
        
        prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger,tb_writer)
                                      

        if opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)
        test_sample = test_sample.to(opt.device)
        _, _, _, crops = model(test_sample)
        img = crops[0].data
        # pic_path = str(opt.result_path)+'/samples/'
        save_img(img, path='/home/zhaoliu/car_brand/racnn/results/samples/iter_{}@2x.jpg'.format(i), annotation=f' 2xstep = {i}')
        # save_img(test_sample, path='/home/zhaoliu/car_brand/racnn/results/samples/iter_{}@1x.jpg'.format(i), annotation=f' 1xstep = {i}')
        # save_img(img, path=f'samples/iter_{i}@2x.jpg', annotation=f'step = {i}')
        # save_img(test_sample, path=f'samples/iter_{i}@4x.jpg', annotation=f'step = {i}')



if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device(f'cuda:3')
    # opt.device = torch.device('cuda')
    cudnn.benchmark = True
    main_worker(opt)