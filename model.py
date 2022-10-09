import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5)) # in_channel=3, out_channel=6, kernel=5*5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), stride=(3, 3))
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dp = nn.Dropout(p=0.2, inplace=False)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = F.relu(self.fc2(x))
        x = self.dp(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=(4, 4), padding=(2, 2))
        self.pool = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(96, 256, (5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(256, 384, (3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 384, (3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(384, 256, (3, 3), padding=(1, 1))
        # self.pool2 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.fc1 = nn.Linear(256*6*6, 256*6*3)
        self.fc2 = nn.Linear(256*6*3, 256*6*3)
        self.fc3 = nn.Linear(256*6*3, 1000)
        self.fc4 = nn.Linear(1000, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 256*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2, dim=1)
        return x





class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=(4, 4), padding=(2, 2))
        self.norm1 = nn.BatchNorm2d(96)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d((3, 3), stride=(2, 2))
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 256, (5, 5), padding=(2, 2))
        self.sw1 = nn.Conv2d(256, 256, (1, 1))
        self.sw2 = nn.Conv2d(256, 256, (1, 1))
        self.sw3 = nn.Conv2d(256, 1, (1, 1))
        # self.fc1 = nn.Linear(256*13*13, 256*13*13)
        self.fc1 = nn.Linear(256*13*13,13*13)
        self.dp = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(13*13, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool(x)
        w = F.relu(self.sw1(x))
        w = F.relu(self.sw2(w))
        w = self.sw3(w)
        # print(w[w>0].shape)
        # w = w/torch.max(w, dim=0)[0]
        x = x*w
        x = x.view(-1, 256*13*13)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        # x = self.dp(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class AttentionInner(nn.Module):
    def __init__(self):
        super(AttentionInner, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=(2, 2)) # 96*55*55
        self.norm1 = nn.BatchNorm2d(96)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(96, 256, (9, 9), padding=(4, 4))
        self.conv3 = nn.Conv2d(256, 384, (9, 9), padding=(4, 4))
        self.conv4 = nn.Conv2d(384, 384, (7, 7), padding=(3, 3))
        self.conv5 = nn.Conv2d(384, 256, (5, 5), padding=(2, 2))
        self.atpool = nn.AvgPool2d((26, 26))
        self.downs = nn.Conv2d(3, 256, (1, 1))
        self.downp = nn.AvgPool2d(())
        self.sw1 = nn.Conv2d(256, 256, (1, 1))
        self.sw2 = nn.Conv2d(256, 256, (1, 1))
        self.sw3 = nn.Conv2d(256, 1, (1, 1))
        self.fc1 = nn.Linear(256 * 26 * 26, 13*13)
        # self.fc2 = nn.Linear(8 * 13 * 13, 8 * 13 * 13)
        self.dp = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(13 * 13, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 96*107*107
        x = self.norm1(x)
        x = self.pool(x) # 96*53*53
        x = F.relu(self.conv2(x)) # 256*53*53
        x = self.norm2(x)
        x = self.pool(x) # 256*26*26
        y = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = y+x
        # x = self.downs(x)

        w = F.relu(self.sw1(x))
        w = F.relu(self.sw2(w))
        w = self.atpool(w)
        x = x * w

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.attention = AttentionInner()
    def forward(self, x):
        x = self.attention(x)
        return x

class Gazer(nn.Module):
    def __init__(self):
        super(Gazer, self).__init__()
        self.fc1 = nn.Linear(256 * 26 * 26, 13 * 13)
        self.dp = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(13 * 13, 3)
    def forward(self, x):
        x = x.view(-1, 256 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 5, padding=2)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 26 * 26, 13*13)
        self.fc2 = nn.Linear(13*13, 1)
        self.bn = nn.BatchNorm2d(256)
        self.dp = nn.Dropout(p=0.5, inplace=False)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn(x)
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = x.view(-1, 256 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        # print(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        # x = torch.abs(F.normalize(x, p=1, dim=1))[:,1]
        # print("mean", x.item())
        return x

class AttentionNet2(nn.Module):
    def __init__(self):
        super(AttentionNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=(2, 2))  # 96*55*55
        self.norm1 = nn.BatchNorm2d(96)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(96, 256, (9, 9), padding=(4, 4))
        self.conv3 = nn.Conv2d(256, 384, (9, 9), padding=(4, 4))
        self.conv4 = nn.Conv2d(384, 384, (7, 7), padding=(3, 3))
        self.conv5 = nn.Conv2d(384, 256, (5, 5), padding=(2, 2))
        self.atpool = nn.AvgPool2d((26, 26))
        self.downs = nn.Conv2d(3, 256, (1, 1))
        self.downp = nn.AvgPool2d(())
        self.sw1 = nn.Conv2d(256, 256, (1, 1))
        self.sw2 = nn.Conv2d(256, 256, (1, 1))
        self.sw3 = nn.Conv2d(256, 1, (1, 1))
        self.fc1 = nn.Linear(256 * 26 * 26, 13 * 13)
        # self.fc2 = nn.Linear(8 * 13 * 13, 8 * 13 * 13)
        self.dp = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(13 * 13, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 96*107*107
        x = self.norm1(x)
        x = self.pool(x)  # 96*53*53
        x = F.relu(self.conv2(x))  # 256*53*53
        x = self.norm2(x)
        x = self.pool(x)  # 256*26*26
        y = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = y + x
        # x = self.downs(x)

        w = F.relu(self.sw1(x))
        w = F.relu(self.sw2(w))
        w = self.atpool(w)
        x = x * w
        x = x.view(-1, 256 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class AttentionNet3(nn.Module):
    def __init__(self):
        super(AttentionNet2, self).__init__()
        self.attention = AttentionInner()
        self.fc1 = nn.Linear(256 * 26 * 26, 13 * 13)
        self.dp = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(13 * 13, 3)

    def forward(self, x):
        x = self.attention(x)
        x = x.view(-1, 256 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x



class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.feature_extractor = torchvision.models.alexnet(pretrained=True).features
        module = getattr(self.feature_extractor, '0')
        module.weight.data = module.weight.data[:, [2, 1, 0]]
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 6 ** 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)
        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv1.bias, val=0.1)
        nn.init.constant_(self.conv2.bias, val=0.1)
        nn.init.constant_(self.conv3.bias, val=1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.005)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.0001)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.0001)
        nn.init.constant_(self.fc1.bias, val=1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        x = x * y
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, x):
        return torch.Tensor([0.08, 0.15, 0.985]).repeat(x.size(0))


class ResnetInner(nn.Module):
    def __init__(self, c, down_sample=True):
        super(ResnetInner, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.conv1 = nn.Conv2d(c, c*2, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(c*2, c*2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c*2)
        self.conv2 = nn.Conv2d(c*2, c*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c*2, c*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(c*2, c*2, kernel_size=3, padding=1)
        self.downs = nn.Conv2d(c, c*2, kernel_size=1, stride=2)

    def forward(self, x):
        res = x
        if self.down_sample:
            res = self.downs(res)
        x = F.relu(self.conv1(x))
        # x = self.bn(x)
        x = F.relu(self.conv2(x))
        # x = self.bn(x)
        x = x+res
        res = x
        x = F.relu(self.conv3(x))
        # x = self.bn(x)
        x = F.relu(self.conv4(x))
        # x = self.bn(x)
        x = x+res
        return x


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResnetInner(32, False)
        self.res2 = ResnetInner(64)
        self.res3 = ResnetInner(128)
        self.res4 = ResnetInner(256)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512, 1000)
        self.dp = nn.Dropout(p=0.2, inplace=False)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        # if self.training:
        #     print("train")
        # else:
        #     print("not train")
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.dp(x)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)

        return x



class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.face = Resnet18()
        self.left = LeNet()
        self.right = LeNet()
        self.fc1 = nn.Linear(9, 10)
        self.fc2 = nn.Linear(10, 3)
        self.dp = nn.Dropout(p=0.5, inplace=False)
    def forward(self, x, left, right):
        x = self.face(x)
        left = self.left(left)
        right = self.right(right)
        y = torch.cat((x, left, right),dim=1)
        # print(y.shape)
        y = y.view(x.size(0),-1)
        y = F.relu(self.fc1(y))
        # y = self.dp(y)
        y = self.fc2(y)
        y = F.normalize(y, p=2, dim=1)
        return y


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.gen = Generator()
        self.classifier1 = nn.Conv2d(256, 64, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(256, 64, kernel_size=(1, 1))
    def forward(self, x):
        x = F.relu(self.gen(x))
        # print(x.size(2))
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2

class Gazer2(nn.Module):
    def __init__(self):
        super(Gazer2, self).__init__()
        self.fc1 = nn.Linear(64 * 26 * 26, 13 * 13)
        self.dp = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(13 * 13, 3)
    def forward(self, x):
        x = x.view(-1, 64 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.layer5(x)
        x2 = self.layer6(x)
        return x1, x2

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        #b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model





