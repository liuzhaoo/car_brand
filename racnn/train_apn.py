import time
from Loss import pairwise_ranking_loss


def apn_train_epoch(epoch,
				data_loader,
				model,
				optimizer,
				device,
				tb_writer):

    print('train_apn at epoch {}'.format(epoch))

    model.train()
    for i,(inputs,targets) in enumerate(data_loader):
		
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()



        t0 = time.time()
        logits, _, _, _ = model(inputs)

        optimizer.zero_grad()
        preds = []
        for j in range(len(targets)):
            pred = [logit[j][targets[j]] for logit in logits]
            preds.append(pred)
        apn_loss = pairwise_ranking_loss(preds)
        apn_loss.backward()
        optimizer.step()
        t1 = time.time()
        itera = (epoch - 1) * int(len(data_loader)) + (i + 1)

        # if (itera % 20) == 0:
        print(" [*] apn_epoch[%d], apn_iter %d || apn_loss: %.4f || Timer: %.4fsec"%(epoch, i, apn_loss.item(), (t1 - t0)))
        tb_writer.add_scalar('train/rank_loss', apn_loss.item(), itera)
     
        


	
