import torch
import torch.nn as nn
import torchexplain
import models.explain_audio_vggish as explain_audio_vggish
import models.explain_video_mobilenet as explain_video_mobilenet


class ExplainAudioVideoMidFusion(nn.Module):
    def __init__(self,
                 nb_classes,
                 train=False,
                 device="cpu"
                ):
        super(ExplainAudioVideoMidFusion, self).__init__()

        if train:
            self.lib = nn
        else:
            self.lib = torchexplain

        vggish_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                      'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                   'releases/download/v0.1/vggish_pca_params-970ea276.pth'
        }

        # -*- Audio model -*-
        # Define audio subnetwork
        audio_model = explain_audio_vggish.VGGish(urls=vggish_urls,
                                          preprocess=False,
                                          postprocess=False,
                                          train=train).to(device)
        self.audio_model = audio_model

        # -*- Video model -*-
        # Define video subnetwork
        video_model = explain_video_mobilenet.get_model(num_classes=600, train=train).to(device)
        # Load weights to video_mobilenet model
        checkpoint = torch.load("models/kinetics_mobilenet_1.0x_RGB_16_best.pth", map_location={"cuda:0":device})
        state_dict = dict()
        for k in list(checkpoint['state_dict'].keys()):
            # each "key" has the word "module." in it.
            # use indexing to remove that.
            state_dict[k[7:]] = checkpoint['state_dict'].pop(k)
        video_model.load_state_dict(state_dict,strict=False)

        # do away with the last layer
        self.video_model = video_model.features

        # Create new classifier
        self.classifier = self.lib.Linear(1024 + 128, nb_classes)

    def forward(self, x_audio, x_video):
        x_audio = self.audio_model(x_audio)
        x_audio = x_audio.view(x_audio.size(0), -1)

        x_video = self.video_model(x_video)

        kernel_size = x_video.data.size()[-3:]
        avg_pool = self.lib.AvgPool3d(kernel_size, stride=kernel_size)
        x_video = avg_pool(x_video).view(-1, 1024)

        x = torch.cat((x_audio, x_video), dim=1)
        return self.classifier(self.lib.ReLU()(x))
