import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def replace_first_conv(res_model, input_ch):
    """
    Adapts the model whose number of input channel is other than 3.
    If the number of input channel is  1 or 4, this function does nothing.
    If the number of input channel is  1 or 4, weights are initialized with pretrained weight.
    Otherwise, weights are initialized with random variables.

    :param res_model:
    :param input_ch:
    :return: model
    """
    if input_ch == 3:
        return res_model

    conv1 = nn.Conv2d(input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    R_IDX = 0
    if input_ch == 1:
        conv1.weight.data = res_model.conv1.weight.data[:, R_IDX:R_IDX + 1, :, :]

    elif input_ch == 4:
        conv1.weight.data[:, :3, :, :] = res_model.conv1.weight.data
        conv1.weight.data[:, 3:4, :, :] = res_model.conv1.weight.data[:, 0:1, :, :]

    res_model.conv1 = conv1
    return res_model


def resnet18(pretrained=False, input_ch=3, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return replace_first_conv(model, input_ch)


def resnet34(pretrained=False, input_ch=3, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return replace_first_conv(model, input_ch)


def resnet50(pretrained=False, input_ch=3, no_replace=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return replace_first_conv(model, input_ch)


def resnet101(pretrained=False, input_ch=3, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return replace_first_conv(model, input_ch)


def resnet152(pretrained=False, input_ch=3, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return replace_first_conv(model, input_ch)


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode="bilinear")
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(.1)

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out


class ResBase(nn.Module):
    def __init__(self, num_classes, layer='50', input_ch=3):
        super(ResBase, self).__init__()

        self.num_classes = num_classes
        print('resnet' + layer)
        if layer == '18':
            resnet = resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            resnet = resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            resnet = resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            resnet = resnet152(pretrained=True, input_ch=input_ch)
        else:
            NotImplementedError

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        img_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        out_dic = {
            "img_size": img_size,
            "conv_x": conv_x,
            "pool_x": pool_x,
            "fm2": fm2,
            "fm3": fm3,
            "fm4": fm4
        }

        return out_dic


class ResClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResClassifier, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)
        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, gen_out_dic):
        gen_out_dic = edict(gen_out_dic)
        fsfm1 = self.fs1(gen_out_dic.fm3, self.upsample1(gen_out_dic.fm4, gen_out_dic.fm3.size()[2:]))
        fsfm2 = self.fs2(gen_out_dic.fm2, self.upsample2(fsfm1, gen_out_dic.fm2.size()[2:]))
        fsfm3 = self.fs4(gen_out_dic.pool_x, self.upsample3(fsfm2, gen_out_dic.pool_x.size()[2:]))
        fsfm4 = self.fs5(gen_out_dic.conv_x, self.upsample4(fsfm3, gen_out_dic.conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, gen_out_dic.img_size)
        out = self.out5(fsfm5)
        return out


def get_models(input_ch, n_class, res="50", is_data_parallel=False):
    model_g = ResBase(n_class, layer=res, input_ch=input_ch)
    model_f1 = ResClassifier(n_class)
    model_f2 = ResClassifier(n_class)
    model_list = [model_g, model_f1, model_f2]

    if is_data_parallel:
        return [torch.nn.DataParallel(x) for x in model_list]
    else:
        return model_list


def get_optimizer(model_parameters, opt, lr, momentum, weight_decay):
    if opt == "sgd":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model_parameters), lr=lr, momentum=momentum,
                               weight_decay=weight_decay)

    elif opt == "adadelta":
        return torch.optim.Adadelta(filter(lambda p: p.requires_grad, model_parameters), lr=lr,
                                    weight_decay=weight_decay)

    elif opt == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model_parameters), lr=lr, betas=[0.5, 0.999],
                                weight_decay=weight_decay)
    else:
        raise NotImplementedError("Only (Momentum) SGD, Adadelta, Adam are supported!")
