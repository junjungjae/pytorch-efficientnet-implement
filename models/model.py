import torch
import torch.nn as nn

import models.modelblock as mb

from torchsummary import summary


def set_block(blocktype, in_channel, out_channel, repeat_n, strides, kernel_size):
    layers = []
    stridelist = [strides] + [1] * (repeat_n - 1)

    for stride in stridelist:
        layers.append(blocktype(in_channel, out_channel, stride, kernel_size))
        in_channel = out_channel

    return_layers = nn.Sequential(*layers)
    print(return_layers)

    return return_layers


class EfficientNet(nn.Module):
    def __init__(self, depth, width, scale, num_classes):
        super(EfficientNet, self).__init__()

        depth_coef = depth
        width_coef = width
        scale_coef = scale

        stage_channel = [32, 16, 24, 40, 80, 112, 192, 320, 1280, 1280]
        compound_stage_channel = [int(x*width_coef) for x in stage_channel]

        layer_repeat = [1, 1, 2, 2, 3, 3, 4, 1, 1]
        compound_layer_repeat = [int(x*depth_coef) for x in layer_repeat]

        stage_stride = [2, 1, 2, 2, 2, 1, 2, 1, 1]
        stage_kernel_size = [3, 3, 3, 5, 3, 5, 5, 3, 1]

        self.scaling = nn.Upsample(scale_factor=scale_coef, mode='bilinear', align_corners=False)

        self.stage1 = set_block(mb.FirstConv, 3, compound_stage_channel[0], compound_layer_repeat[0],
                                stage_stride[0], stage_kernel_size[0])

        self.stage2 = set_block(mb.NormConv, compound_stage_channel[0], compound_stage_channel[1], compound_layer_repeat[1],
                                stage_stride[1], stage_kernel_size[1])

        self.stage3 = set_block(mb.MBConv, compound_stage_channel[1], compound_stage_channel[2], compound_layer_repeat[2],
                                stage_stride[2], stage_kernel_size[2])

        self.stage4 = set_block(mb.MBConv, compound_stage_channel[2], compound_stage_channel[3], compound_layer_repeat[3],
                                stage_stride[3], stage_kernel_size[3])

        self.stage5 = set_block(mb.MBConv, compound_stage_channel[3], compound_stage_channel[4], compound_layer_repeat[4],
                                stage_stride[4], stage_kernel_size[4])

        self.stage6 = set_block(mb.MBConv, compound_stage_channel[4], compound_stage_channel[5], compound_layer_repeat[5],
                                stage_stride[5], stage_kernel_size[5])

        self.stage7 = set_block(mb.MBConv, compound_stage_channel[5], compound_stage_channel[6], compound_layer_repeat[6],
                                stage_stride[6], stage_kernel_size[6])

        self.stage8 = set_block(mb.MBConv, compound_stage_channel[6], compound_stage_channel[7], compound_layer_repeat[7],
                                stage_stride[7], stage_kernel_size[7])

        self.stage9 = set_block(mb.LastConv, compound_stage_channel[7], compound_stage_channel[8], compound_layer_repeat[8],
                                stage_stride[8], stage_kernel_size[8])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc_layer = nn.Linear(compound_stage_channel[8], num_classes)

    def forward(self, x):
        out = self.scaling(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.stage8(out)
        out = self.stage9(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc_layer(out)

        return out




