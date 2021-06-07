import torchstat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .resnet import resnet18
device = torch.device(f'cuda:1')
# device = torch.device(f'cuda:4')
class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        def h(_x): return 1 / (1 + torch.exp(-10 * _x.float()))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.to(device), y.to(device)

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]/2
            tl = tl if tl > (in_size/3) else in_size/3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, w_off: w_end, h_off: h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(224, 224), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size/3*2)
        short_size = (in_size/3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size)+(x >= long_size)+(y < short_size)+(y >= long_size)) > 0).float()*2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.to(device)
            my_batch = my_batch.to(device)
            ml_batch = ml_batch.to(device)
            ret = ret.to(device)

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret

class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)

class RACNN(nn.Module):
    def __init__(self, num_classes, img_scale=448):
        super(RACNN, self).__init__()

        self.b1 = resnet18(num_classes = num_classes)
        self.b2 = resnet18(num_classes = num_classes)
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)

        self.feature_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)  # 分类
        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)        # apn
        self.crop_resize = AttentionCropLayer()                        # apn之后用于裁剪
        self.apn = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )

        
    def forward(self,x):
        # batch_size = x.shape[0]
        # rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False)
        # forward @scale-1
        feature_s1 = self.b1(x)  # torch.Size([1, 512, 8, 8])          # resnet的输出
        pool_s1 = self.feature_pool(feature_s1)                        # resnet的输出 进行池化，([1,512,1,1])
        attention_s1 = self.apn(feature_s1.view(-1, 512 * 8 * 8))    # apn的输出
        # attention_s1 = _attention_s1*rescale_tl                        # 三个坐标
        resized_s1 = self.crop_resize(x, attention_s1 * x.shape[-1])   # 在原图上进行裁剪
        # forward @scale-2
        feature_s2 = self.b1(resized_s1)                               # 裁剪之后的图经过resnet
        pool_s2 = self.feature_pool(feature_s2)                 
             
        pred1 = self.classifier1(pool_s1.view(-1, 512))                # 对scale1中的resnet结果分类
        pred2 = self.classifier2(pool_s2.view(-1, 512))                # 对scale1中的resnet结果分类

        return [pred1, pred2], [feature_s1, feature_s2], [attention_s1], [resized_s1]

if __name__ == "__main__":
    net = RACNN(num_classes=1189)
    # net.mode('pretrain_apn')
    # optimizer = torch.optim.SGD(list(net.apn1.parameters()) + list(net.apn2.parameters()), lr=0.001, momentum=0.9)
    # for i in range(50):
    inputs = torch.rand(2, 3, 224, 224)
    inputs = Variable(inputs)

    y = net(inputs)

    print(len(y))