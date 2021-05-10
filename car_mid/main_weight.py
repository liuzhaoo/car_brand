
import json
import random
import os
import torchvision
import torch

import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist


from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler, Adam
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize,RandomRotation,RandomCrop,RandomHorizontalFlip
from pathlib import Path
from model.inceptionv4 import Inceptionv4  
from model.resnet import resnet18
from opts import parse_opts
from utils import get_mean_std,Logger, worker_init_fn, get_lr
from dataset.train_dataloader import LmdbDataset_train
from dataset.val_dataloader import LmdbDataset_val
# from dataset.train_dataloader_txt import TxtDataset_train
# from dataset.val_dataloader_txt import TxtDataset_val

from training import train_epoch
from validation import val_epoch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def save_checkpoint(save_file_path, epoch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

def genarate_model(opt):
    if opt.model == 'inceptionv4':

        model = Inceptionv4(classes=opt.fb_cls)
    elif opt.model == 'resnet18':
        model = resnet18(num_classes=opt.fb_cls)
    return model

def make_data_parallel(model, device):

    if device.type == 'cuda' and device.index is not None:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
            # device = torch.device("cuda", local_rank)
            # torch.cuda.set_device(device)
        model.to(device)

        model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
    else:
        # model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    return model

def get_opt():
    opt = parse_opts()
    opt.begin_epoch = 1
    opt.n_input_channels = 3
    opt.result_path = Path('/home/zhaoliu/car_brand/car_mid/results/weight_sample_s')
    if opt.distributed:
        opt.dist_rank = 0  # 单个机器
        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt

# def make_weights_for_balanced_classes(images, nclasses):                        
#     count = [0] * nclasses                                                      
#     for item in images:                                                         
#         count[item[1]] += 1        # 统计每类的数量
#     print('数量统计完成')                                          
#     weight_per_class = [0.] * nclasses                                      
#     N = float(sum(count))          # 计算所有样本总数                                      
#     for i in range(nclasses):                                                   
#         weight_per_class[i] = N/float(count[i])                                 
#     weight = [0] * len(images)                                              
#     for idx, val in enumerate(images):                                          
#         weight[idx] = weight_per_class[val[1]]                                  
#     return weight      


def make_label_list(txt1,txt2):
    with open(txt1,'r') as f:
        lines = f.readlines()

def make_weights_for_balanced_classes(txt, nclasses):          
    with open(txt,'r') as f:
        lines = f.readlines()

    count = [0] * nclasses                                                      
    for item in lines: 
        
        label = int(item.strip().split('\t')[1])
        count[label] += 1        # 统计每类的数量
                                                        
               
    print('数量统计完成')                                       
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))          # 计算所有样本总数                                      
    for i in range(nclasses): 
        try:                                         
            weight_per_class[i] = N/float(count[i])  # 每一类的权重
        except:
            continue
          
                                   
    weight = [0] * len(lines)    
         
                                     
    for idx, val in enumerate(lines): 
        
        label = int(val.strip().split('\t')[1])  
                                        
        weight[idx] = weight_per_class[label] 
        # if idx % 50000 == 0:
        #     print(idx)                            
    return weight       


def get_train_utils(opt, model_parameters):

    transform = []
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    resize = Resize(opt.size)



    transform.append(RandomHorizontalFlip())
    transform.append(RandomCrop(opt.size))
    transform.append(resize)
    
    # transform.append(RandomRotation(45))
    transform.append(ToTensor())
    transform.append(normalize)
    
    
    transform = Compose(transform)

    training_data = LmdbDataset_train(opt.train_data,transform,opt.keys_path_train)

    # print('均衡化。。。')

    # # weights = make_weights_for_balanced_classes('/mnt/disk/zhaoliu_data/carlogo/train_data/carlogo_train.txt', 27249)

    weights = make_weights_for_balanced_classes(opt.weight_txt, 1507)
    # np.save('/home/zhaoliu/car_weight/car_full/dataset/weights.npy',weights)

    weights = torch.DoubleTensor(weights)
    print(len(training_data))
    print(len(weights))
    assert len(training_data) == len(weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    print('均衡化完成')


    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            training_data)

    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=opt.batch_size,
                                               shuffle=not train_sampler,
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn,
                                               sampler=train_sampler)
    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    if opt.optimizer == 'sgd':
        optimizer = SGD(model_parameters,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        dampening=dampening,
                        weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = Adam(model_parameters,
                        lr=opt.learning_rate)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    
    transform = []
    # normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
    #                                  opt.no_std_norm)
    
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    resize = Resize(opt.size)
    # transform.append(ToPILImage())
    transform.append(resize)

    transform.append(ToTensor())
    transform.append(normalize)
    transform = Compose(transform)
    
    val_data = LmdbDataset_val(opt.val_data,transform,opt.keys_path_val)

    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=opt.batch_size,
                                             shuffle=not val_sampler,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn,
                                             sampler=val_sampler)
                                            #  worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger

def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        # opt.n_threads = int(
        #     (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0


    model = genarate_model(opt)     
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if opt.distributed:
        model = make_data_parallel(model,opt.device)
    else:
        model.to(opt.device)
        # model = nn.DataParallel(model).cuda()

    print('Total params: %.2fM' % (sum(p.numel()
                                       for p in model.parameters()) / 1000000.0))
    if opt.is_master_node:
        print(model)
    parameters = model.parameters()
    criterion = CrossEntropyLoss().to(opt.device)

    (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)

    val_loader, val_logger = get_val_utils(opt)

    if not opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    print('数据加载完毕')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
                # train_sampler2.set_epoch(i)
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, opt.is_master_node, tb_writer, opt.distributed)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i,model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger,opt.is_master_node, tb_writer,
                                      opt.distributed)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    # if opt.inference:
    #     inference_loader, inference_class_names = get_inference_utils(opt)
    #     inference_result_path = opt.result_path / '{}.json'.format(
    #         opt.inference_subset)

    #     inference.inference(inference_loader, model, inference_result_path,
    #                         inference_class_names, opt.inference_no_average,
    #                         opt.output_topk)

if __name__ == '__main__':
    opt = get_opt()
    # torch.multiprocessing.set_start_method('spawn')
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    torchvision.set_image_backend('accimage')
    # if opt.accimage:
    #     torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()

    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    
    else:
        main_worker(-1, opt)