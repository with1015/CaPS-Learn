import torch
import torch.nn as nn


class VGG16(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
                                        nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_classes)
                                        )


    def forward(self, x):
        result = self.features(x)
        result = self.avgpool(result)
        result = torch.flatten(result, 1)
        result = self.classifier(result)
        return result


    def _make_layers(self):
        layers = []
        in_channels = 3
        for cfg in self.config:
            if cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                            nn.BatchNorm2d(cfg),
                            nn.ReLU(inplace=True)]
                in_channels = cfg
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
