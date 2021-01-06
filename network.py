from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tvm


class CustomBlock(nn.Module):

    def __init__(self, blocks=3, hidden=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('bn0', nn.BatchNorm1d(hidden)),
                ('fc0', nn.Linear(hidden, hidden)),
                ('rl0', nn.ReLU()),
                #('do0', nn.Dropout(0.25)),
            ]))
            for _ in range(blocks)
        ])

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            y = block(x)
            if i == 1:
                F.dropout(y, p=0.25, training=self.training, inplace=True)
            x = y + x
        return x


class CNNnetwork(nn.Module):

    def __init__(self, out_dim=13, mode="PER_IMAGE", input_channel="GRAY"):
        super().__init__()
        self.mode = mode
        if input_channel == "RGB":
            num_hidden = 1000
            self.main = tvm.densenet121(pretrained=True)
            self.custom = nn.Sequential(OrderedDict([
                ('fc0', nn.Linear(num_hidden, out_dim)),
            ]))
        else:
            num_hidden = 256
            self.main = tvm.densenet169(pretrained=False, drop_rate=0.25, num_classes=num_hidden)
            self.main.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

            self.custom = nn.Sequential(OrderedDict([
                ('cb0', CustomBlock(blocks=3, hidden=num_hidden)),
                ('fc0', nn.Linear(num_hidden, out_dim)),
            ]))


    def forward(self, x, num_chs):
        if self.mode == "PER_IMAGE":
            z = self.main(x)
        else:
            y = [x[b, :c, :, :] for b, c in enumerate(num_chs)]
            y = torch.cat(y).unsqueeze(dim=1)
            y = self.main(y)
            y = F.softmax(y, dim=1)
            z = torch.split(y, num_chs.tolist(), dim=0)
            z = torch.cat([t.mean(dim=0, keepdim=True) for t in z])
        x = self.custom(z)
        return x

class ClassifierWrapper(CNNnetwork):
    def forward(self, batch):
        preds = super().forward(x=batch.image, num_chs=batch.channels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        losses = self.loss(preds, batch.target).mean(dim=0)
        loss = losses.mean()

        return {"loss": loss, "preds": preds}


if __name__ == "__main__":
    m = Network()
    m.to_distributed("cuda:0")