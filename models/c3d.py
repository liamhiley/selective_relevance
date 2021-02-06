import torch
import torch.nn as nn
from .. import torchexplain


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False, range=(-1,1), embedding=False, train=True):
        super(C3D, self).__init__()

        if train:
            self.lib = torch.nn
        else:
            self.lib = torchexplain
        self.layer1 = nn.Sequential(
            self.lib.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.layer2 = nn.Sequential(
            self.lib.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.layer3 = nn.Sequential(
            self.lib.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.layer4 = nn.Sequential(
            self.lib.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.layer5 = nn.Sequential(
            self.lib.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            self.lib.ReLU(inplace=True),
            self.lib.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )

        self.fc6 = self.lib.Linear(8192, 4096)
        self.fc7 = self.lib.Linear(4096, 4096)
        if not embedding:
            self.fc8 = self.lib.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = self.lib.ReLU()

        self.embedding = embedding

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        if self.embedding:
            return self.fc7(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)


        return logits

    def load_pretrained_weights(self, model_dir):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(model_dir)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, self.lib.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model, prefix=""):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    if prefix == "module":
        model = model.module
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
def c3d(num_classes, sample_size, sample_duration):
    return C3D(num_classes)

class C3DasVGG(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False, range=(-1,1), embedding=False, train=False):
        super(C3DasVGG, self).__init__()

        if train:
            self.lib = torch.nn
        else:
            self.lib = torchexplain

        self.conv1 = self.lib.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1),range=range)
        self.pool_spatial = self.lib.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = self.lib.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_uniform = self.lib.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = self.lib.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = self.lib.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv4a = self.lib.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = self.lib.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv5a = self.lib.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = self.lib.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_padded = self.lib.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = self.lib.Linear(8192, 4096)
        self.fc7 = self.lib.Linear(4096, 4096)
        self.fc8 = self.lib.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = self.lib.ReLU(inplace=True)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool_spatial,
            self.conv2,
            self.relu,
            self.pool_uniform,
            self.conv3a,
            self.relu,
            self.conv3b,
            self.relu,
            self.pool_uniform,
            self.conv4a,
            self.relu,
            self.conv4b,
            self.relu,
            self.pool_uniform,
            self.conv5a,
            self.relu,
            self.conv5b,
            self.relu,
            self.pool_padded
        )

        self.dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Sequential(
                self.fc6,
                self.relu,
                self.dropout,
                self.fc7,
                self.relu,
                self.dropout,
                self.fc8
        )

        self.embedding = embedding

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.features(x)
        x = x.view(-1, 8192)

        logits = self.classifier(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, self.lib.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())
