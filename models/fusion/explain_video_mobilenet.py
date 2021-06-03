'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
https://github.com/okankop/Efficient-3DCNNs/blob/master/models/mobilenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ... import torchexplain
import pdb
lib = torchexplain

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        lib.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False,range=(-2.9614062262808107,3.887405778629379)),
        lib.BatchNorm3d(oup),
        lib.ReLU(inplace=True)
    )


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,train=False):
        super(Block, self).__init__()
        self.conv1 = lib.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = lib.BatchNorm3d(in_planes)
        self.conv2 = lib.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = lib.BatchNorm3d(out_planes)

    def forward(self, x):
        out = lib.ReLU()(self.bn1(self.conv1(x)))
        out = lib.ReLU()(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=600, sample_size=224, width_mult=1., train=False):
        super(MobileNet, self).__init__()
        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
            # c, n, s
            [64,   1, (2,2,2)],
            [128,  2, (2,2,2)],
            [256,  2, (2,2,2)],
            [512,  6, (2,2,2)],
            [1024, 2, (1,1,1)],
        ]
        self.first = conv_bn(3, input_channel, (1,2,2))
        self.features = [conv_bn(3, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            lib.Linear(last_channel, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = lib.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet(**kwargs)
    return model


if __name__ == '__main__':
    model = get_model(num_classes=600, sample_size=112, width_mult=1.)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)
