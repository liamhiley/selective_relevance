import torch
import torch.nn as nn
from ... import torchexplain
from . import explain_audio_vggish as explain_audio_vggish
from . import explain_video_mobilenet as explain_video_mobilenet


class ExplainAudioVideoMidFusion(nn.Module):
    def __init__(self,
                 nb_classes,
                 range=None,
                 train=False,
                 device="cpu"
                ):
        super(ExplainAudioVideoMidFusion, self).__init__()

        if train:
            self.lib = nn
        else:
            self.lib = torchexplain

        # -*- Audio model -*-
        # Define audio subnetwork
        audio_model = explain_audio_vggish.VGGish(urls=None,
                                          preprocess=False,
                                          postprocess=False,
                                          train=train).to(device)
        self.audio_model = audio_model

        # -*- Video model -*-
        # Define video subnetwork
        video_model = explain_video_mobilenet.get_model(num_classes=600, train=train, range=range).to(device)

        # do away with the last layer
        self.video_model = video_model.features

        # Create new classifier
        self.classifier = self.lib.Linear(1024 + 128, nb_classes)

    def forward(self, inp):
        x_audio, x_video = inp
        x_audio = self.audio_model(x_audio)
        x_audio = x_audio.view(x_audio.size(0), -1)

        x_video = self.video_model(x_video)

        kernel_size = x_video.data.size()[-3:]
        avg_pool = self.lib.AvgPool3d(kernel_size, stride=kernel_size)
        x_video = avg_pool(x_video).view(-1, 1024)

        x = torch.cat((x_audio, x_video), dim=1)
        return self.classifier(self.lib.ReLU()(x))

def generate_model(
    num_classes,
    range=range,
    train=False,
    device='cpu',
    **kwargs
):
    return ExplainAudioVideoMidFusion(num_classes,range,train,device)
