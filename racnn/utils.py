import matplotlib.pyplot as plt
import numpy as np
import torch
import csv

def save_img(x, path, annotation=''):
    fig = plt.gcf()  # generate outputs
    plt.imshow(tensor_to_img(x[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image

def tensor_to_img(x, imtype=np.uint8):
	mean = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
	std = [1. / 255., 1. / 255., 1. / 255.]

	if not isinstance(x, np.ndarray):
		if isinstance(x, torch.Tensor):  # get the data from a variable
			image_tensor = x.data
		else:
			return x
		image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
		if image_numpy.shape[0] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (3, 1, 1))
		for i in range(len(mean)):
			image_numpy[i] = image_numpy[i] * std[i] + mean[i]
		image_numpy = image_numpy * 255
		image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
	else:  # if it is a numpy array, do nothing
		image_numpy = x
	return image_numpy.astype(imtype)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size
def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)
