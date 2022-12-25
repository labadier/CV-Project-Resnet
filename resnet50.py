import torch

class ResBottleneckBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = torch.nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.shortcut = torch.nn.Sequential()
        
        if self.downsample or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                torch.nn.BatchNorm2d(out_channels)
            )

        self.bn1 = torch.nn.BatchNorm2d(out_channels//4)
        self.bn2 = torch.nn.BatchNorm2d(out_channels//4)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = torch.nn.ReLU()(self.bn1(self.conv1(input)))
        input = torch.nn.ReLU()(self.bn2(self.conv2(input)))
        input = torch.nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return torch.nn.ReLU()(input)


class ResNet50(torch.nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        
        filters = [64, 256, 512, 1024, 2048]
        repeat = [3, 4, 6, 3]

        self.layer1 = torch.nn.Sequential()
        self.layer1.add_module('conv2_1', ResBottleneckBlock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module(f'conv2_{i+1}', ResBottleneckBlock(filters[1], filters[1], downsample=False))

        self.layer2 = torch.nn.Sequential()
        self.layer2.add_module('conv3_1', ResBottleneckBlock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module(f'conv3_{i+1}', ResBottleneckBlock(filters[2], filters[2], downsample=False))

        self.layer3 = torch.nn.Sequential()
        self.layer3.add_module('conv4_1', ResBottleneckBlock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module(f'conv4_{i+1}', ResBottleneckBlock(filters[3], filters[3], downsample=False))

        self.layer4 = torch.nn.Sequential()
        self.layer4.add_module('conv5_1', ResBottleneckBlock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module(f'conv5_{i+1}', ResBottleneckBlock(filters[4], filters[4], downsample=False))

        self.gap = torch.torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.torch.nn.Linear(filters[4], 10)

        self.loss_criterion = torch.nn.CrossEntropyLoss() 

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)
        print('ResNet50 model created. Device:', self.device)


    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)

        return input


    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def makeOptimizer(self, lr, decay=1e-4):

        # opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience=5)
        opt = torch.optim.SGD(self.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        return opt, scheduler

    def computeLoss(self, outputs, data):
        return self.loss_criterion(outputs, data['labels'].to(self.device) )