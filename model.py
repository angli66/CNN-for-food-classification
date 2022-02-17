import torch
import torch.nn as nn
import torchvision.models as models

class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) # 3 input image channels (RGB), 64 output channels, 3x3 convolution kernel
        self.bn1 = nn.BatchNorm2d(64) # 64 input channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 20)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(inplace=True)

        # Xavier initialization
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class custom(nn.Module):
    def __init__(self):
        super(custom, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) # 3 input image channels (RGB), 64 output channels, 3x3 convolution kernel
        self.bn1 = nn.BatchNorm2d(64) # 64 input channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 20)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(inplace=True)

        # Xavier initialization
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def get_model(args):
    if args['model'] == 'baseline':
        model = baseline()

    elif args['model'] == 'custom':
        model = custom()

    elif args['model'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 20)
        nn.init.kaiming_normal_(model.fc.weight)

        if args['pt_ft'] == True:
            for name, param in model.named_parameters():
                if not 'fc' in name:
                    param.requires_grad = False
    
    elif args['model'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, 20)
        nn.init.kaiming_normal_(model.classifier[-1].weight)

        if args['pt_ft'] == True:
            for name, param in model.named_parameters():
                if not 'classifier.6' in name:
                    param.requires_grad = False

    else:
        raise NotImplementedError(f"get_model not implemented for model {args}")
    
    return model
