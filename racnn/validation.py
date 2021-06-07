import time
import torch
import torch.distributed as dist
import numpy as np
from utils import AverageMeter, calculate_accuracy,get_lr
from torch.utils.data.dataloader import default_collate


def val_epoch(epoch,
				data_loader,
				model,
				criterion,
				device,
				epoch_logger,
				tb_writer=None):

    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    accuracies_cls = AverageMeter()
    accuracies_apn = AverageMeter()
    end_time = time.time()

    with torch.no_grad():
        for i,(inputs,targets) in enumerate(data_loader):

            data_time.update(time.time() - end_time)
            
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs, _, _, _ = model(inputs)

            loss1 = criterion(outputs[0], targets)
            loss2 = criterion(outputs[1], targets)
            loss = loss1 + loss2
            
            acc1= calculate_accuracy(outputs[0], targets)
            acc2= calculate_accuracy(outputs[1], targets)

            losses.update(loss.item(), inputs.size(0))
            losses1.update(loss1.item(), inputs.size(0))
            losses2.update(loss2.item(), inputs.size(0))
            accuracies_cls.update(acc1, inputs.size(0))
            accuracies_apn.update(acc2, inputs.size(0))

            

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            itera = (epoch - 1) * int(len(data_loader)) + (i + 1)



            if tb_writer is not None:
                    
                tb_writer.add_scalar('val_iter/loss_iter', losses.val, itera)
                tb_writer.add_scalar('val_iter/loss_cls_iter', losses1.val, itera)
                tb_writer.add_scalar('val_iter/loss_apn_iter', losses2.val, itera)
                tb_writer.add_scalar('val_iter/acc_cls_iter', accuracies_cls.val, itera)
                tb_writer.add_scalar('val_iter/acc_apn_iter', accuracies_apn.val, itera)

                

            print('val Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Loss_cls {loss1.val:.4f} ({loss1.avg:.4f})\t'
                    'Loss_apn {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    'Acc_cls {acc1.val:.3f} ({acc1.avg:.3f})\t'
                    'Acc_apn {acc2.val:.3f} ({acc2.avg:.3f})'.format(epoch,
                                        i + 1,
                                        len(data_loader),
                                        batch_time=batch_time,
                                        data_time=data_time,
                                        loss=losses,
                                        loss1=losses1,
                                        loss2=losses2,
                                        acc1=accuracies_cls,
                                        acc2=accuracies_apn))
         

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
			'loss_cls': losses1.avg,
			'loss_apn': losses2.avg,
            'acc_cls': accuracies_cls.avg,
            'acc_apn': accuracies_apn.avg})
    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/loss_cls', losses1.avg, epoch)
        tb_writer.add_scalar('val/loss_apn', losses2.avg, epoch)
        tb_writer.add_scalar('val/acc_cls', accuracies_cls.avg, epoch)
        tb_writer.add_scalar('val/acc_apn', accuracies_apn.avg, epoch)
