import time
import torch.distributed as dist
import numpy as np
from utils import AverageMeter, calculate_accuracy,get_lr
from torch.utils.data.dataloader import default_collate


def cls_train_epoch(epoch,
				data_loader,
				model,
				criterion,
				optimizer,
				device,
				epoch_logger,
				tb_writer=None):

	print('train at epoch {}'.format(epoch))

	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	losses1 = AverageMeter()
	losses2 = AverageMeter()
	accuracies1 = AverageMeter()
	accuracies2 = AverageMeter()
	end_time = time.time()

	for i,(inputs,targets) in enumerate(data_loader):

		data_time.update(time.time() - end_time)
		
		inputs = inputs.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad()
		
		outputs, _, _, _ = model(inputs)

		loss1 = criterion(outputs[0], targets)
		loss2 = criterion(outputs[1], targets)
		loss = loss1 + loss2
		
		acc1= calculate_accuracy(outputs[0], targets)
		acc2= calculate_accuracy(outputs[1], targets)

	
		losses.update(loss.item(), inputs.size(0))
		losses1.update(loss1.item(), inputs.size(0))
		losses2.update(loss2.item(), inputs.size(0))
		accuracies1.update(acc1, inputs.size(0))
		accuracies2.update(acc2, inputs.size(0))

		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end_time)
		end_time = time.time()
		itera = (epoch - 1) * int(len(data_loader)) + (i + 1)
		batch_lr=get_lr(optimizer)


		if tb_writer is not None:
				

			tb_writer.add_scalar('train_iter/loss_iter', losses.val, itera)
			tb_writer.add_scalar('train_iter/loss_cls_iter', losses1.val, itera)
			tb_writer.add_scalar('train_iter/loss_apn_iter', losses2.val, itera)
			tb_writer.add_scalar('train_iter/acc_cls_iter', accuracies1.val, itera)
			tb_writer.add_scalar('train_iter/acc_apn_iter', accuracies2.val, itera)
			tb_writer.add_scalar('train_iter/lr_iter', batch_lr, itera)
			

		print('Train Epoch: [{0}][{1}/{2}]\t'
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
									   acc1=accuracies1,
									   acc2=accuracies2))


	if epoch_logger is not None:
		epoch_logger.log({
			'epoch': epoch,
			'loss': losses.avg,
			'loss_cls': losses1.avg,
			'loss_apn': losses2.avg,
			'acc_cls': accuracies1.avg,
			'acc_apn': accuracies2.avg,
			'lr': batch_lr})

	if tb_writer is not None:
		tb_writer.add_scalar('train/loss', losses.avg, epoch)
		tb_writer.add_scalar('train/loss_cls', losses1.avg, epoch)
		tb_writer.add_scalar('train/loss_apn', losses2.avg, epoch)
		tb_writer.add_scalar('train/acc_cls', accuracies1.avg, epoch)
		tb_writer.add_scalar('train/acc_apn', accuracies2.avg, epoch)
		tb_writer.add_scalar('train/lr', batch_lr, epoch)
