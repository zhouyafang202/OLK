import torch
import torch.nn as nn
from torchvision import models

from models.VGG_backone import vgg
import torch.nn.functional as F
from torch.nn import init
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

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

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnet34_CA(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet_CA(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)



def resnet50_CA(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet_CA(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
def resnet50_CBAM(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet_CBAM(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101_CA(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet_CA(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d_CA(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet_CA(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d_CA(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
# =========================
# 定义模型
# =========================
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.attn = None

    def forward(self, x):
        # x.shape = (batch_size, seq_len, embed_size)

        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)  # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_scores = attention_scores / (self.embed_size ** 0.5)  # Scale the scores

        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Apply the attention weights to the value matrix
        out = torch.bmm(attention_probs, V)  # (batch_size, seq_len, embed_size)

        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadedAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 确保embed_size能够被num_heads整除
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"

        # 定义多个头的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # 输出线性变换
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 1. 计算查询、键和值的映射
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)  # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        # 2. 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. 转置后进行batch matrix multiplication (BMM)
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / (self.head_dim ** 0.5)  # Scale the scores

        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 4. 计算加权值
        out = torch.matmul(attention_probs, V)  # (batch_size, num_heads, seq_len, head_dim)

        # 5. 拼接所有头的输出
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.num_heads * self.head_dim)  # (batch_size, seq_len, embed_size)

        # 6. 输出线性变换
        out = self.fc_out(out)  # (batch_size, seq_len, embed_size)

        return out





class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1,1,0, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1,1,0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], dim=1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out



class ResNet_CBAM(nn.Module):

    def __init__(self, num_grades, num_features, embed_dim=256, num_heads=4, dropout_rate=0.5):
        super(ResNet_CBAM, self).__init__()
        resnet = resnet50_CBAM(include_top=False)


        modules = list(resnet.children())  # 去掉平均池化层和全连接层
        self.image_model = nn.Sequential(*modules)  # 得到特征图

        self.conv1x1 = nn.Conv2d(2048, embed_dim, kernel_size=1)  # 将通道数转换为 embed_dim
        self.dropout_conv = nn.Dropout(p=dropout_rate)  # 在卷积层之后添加 Dropout
        # self.layer_norm = nn.LayerNorm(embed_dim)  # 添加层归一化
        # 结构化特征的全连接层
        self.feature_fc = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, embed_dim)
            # nn.ReLU(),
            # nn.Dropout(p=dropout_rate)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc_dysplasia = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embed_dim, num_grades)
        )

        self.fc_cancer = nn.Sequential(
            nn.Linear(embed_dim * 2 + num_grades, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embed_dim , embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embed_dim // 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, image, features):
        batch_size = image.size(0)

        image_features = self.image_model(image)
        image_features = self.conv1x1(image_features)
        image_features = self.avgpool(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        features_encoded = self.feature_fc(features)
        attn_output = torch.cat([image_features, features_encoded], dim=1)
        # output
        dysplasia_output = self.fc_dysplasia(attn_output)
        # feature_fusion
        combined_features = torch.cat([attn_output, dysplasia_output], dim=1)
        cancer_output = self.fc_cancer(combined_features)
        return cancer_output, dysplasia_output