import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AMCNet(nn.Module):
    def __init__(self):
        super(AMCNet, self).__init__()
        ################################resnet101 Flow#######################################
        feats_Flow = models.resnet101(pretrained=True)
        self.conv0_Flow = nn.Sequential(feats_Flow.conv1, feats_Flow.bn1, nn.PReLU())
        self.conv1_Flow = nn.Sequential(feats_Flow.maxpool, *feats_Flow.layer1)
        self.conv2_Flow = feats_Flow.layer2
        self.conv3_Flow = feats_Flow.layer3
        self.conv4_Flow = feats_Flow.layer4

        ################################resnet101 RGB#######################################
        feats_RGB = models.resnet101(pretrained=True)
        self.conv0_RGB = nn.Sequential(feats_RGB.conv1, feats_RGB.bn1, nn.PReLU())
        self.conv1_RGB = nn.Sequential(feats_RGB.maxpool, *feats_RGB.layer1)
        self.conv2_RGB = feats_RGB.layer2
        self.conv3_RGB = feats_RGB.layer3
        self.conv4_RGB = feats_RGB.layer4

        self.atten_flow_channel_0 = ChannelAttention(64)
        self.atten_flow_channel_1 = ChannelAttention(256)
        self.atten_flow_channel_2 = ChannelAttention(512)
        self.atten_flow_channel_3 = ChannelAttention(1024)
        self.atten_flow_channel_4 = ChannelAttention(2048)

        self.atten_flow_spatial_0 = SpatialAttention()
        self.atten_flow_spatial_1 = SpatialAttention()
        self.atten_flow_spatial_2 = SpatialAttention()
        self.atten_flow_spatial_3 = SpatialAttention()
        self.atten_flow_spatial_4 = SpatialAttention()

        self.attention_feature0 = nn.Sequential(nn.Conv2d(64*2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                                nn.PReLU(),
                                                nn.Conv2d(32, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(256*2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(512*2, 128, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(128), nn.PReLU(),
                                                nn.Conv2d(128, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(1024*2, 256, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(256), nn.PReLU(),
                                                nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(2048*2, 512, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(512), nn.PReLU(),
                                                nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                                nn.PReLU(),
                                                nn.Conv2d(128, 2, kernel_size=3, padding=1))

        self.gate_RGB4 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.gate_RGB3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.gate_RGB2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.gate_RGB1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.gate_RGB0 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())

        self.gate_Flow4 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.gate_Flow3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.gate_Flow2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.gate_Flow1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.gate_Flow0 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())

        self.fuse4_Flow = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_RGB = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.channel4 = ChannelAttention(512)
        self.spatial4 = SpatialAttention()

        self.fuse3_Flow = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_RGB = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.channel3 = ChannelAttention(256)
        self.spatial3 = SpatialAttention()

        self.fuse2_Flow = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_RGB = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.channel2 = ChannelAttention(128)
        self.spatial2 = SpatialAttention()

        self.fuse1_Flow = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_RGB = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.channel1 = ChannelAttention(64)
        self.spatial1 = SpatialAttention()

        self.fuse0_Flow = nn.Sequential(nn.Conv2d(32*2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.fuse0_RGB = nn.Sequential(nn.Conv2d(32*2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.channel0 = ChannelAttention(32)
        self.spatial0 = SpatialAttention()
        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                     nn.Conv2d(32, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, input, flow):
        c0_Flow = self.conv0_Flow(flow)  # N,64,192,192
        c1_Flow = self.conv1_Flow(c0_Flow)  # N,256,96,96
        c2_Flow = self.conv2_Flow(c1_Flow)  # N,512,48,48
        c3_Flow = self.conv3_Flow(c2_Flow)  # N,1024,24,24
        c4_Flow = self.conv4_Flow(c3_Flow)  # N,2048,12,12

        c0_RGB = self.conv0_RGB(input)  # N,64,192,192
        G0 = self.attention_feature0(torch.cat((c0_RGB, c0_Flow), dim=1))
        G0 = F.adaptive_avg_pool2d(torch.sigmoid(G0), 1)
        c0_RGB = G0[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * c0_RGB
        c0_Flow = G0[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * c0_Flow
        temp = c0_Flow.mul(self.atten_flow_channel_0(c0_Flow))
        temp = temp.mul(self.atten_flow_spatial_0(temp))
        c0_RGB = c0_RGB+temp

        c1_RGB = self.conv1_RGB(c0_RGB)  # N,256,96,96
        G1 = self.attention_feature1(torch.cat((c1_RGB, c1_Flow), dim=1))
        G1 = F.adaptive_avg_pool2d(torch.sigmoid(G1), 1)
        c1_RGB = G1[:, 0, :, :].unsqueeze(1).repeat(1, 256, 1, 1) * c1_RGB
        c1_Flow = G1[:, 1, :, :].unsqueeze(1).repeat(1, 256, 1, 1) * c1_Flow
        temp = c1_Flow.mul(self.atten_flow_channel_1(c1_Flow))
        temp = temp.mul(self.atten_flow_spatial_1(temp))
        c1_RGB = c1_RGB+temp

        c2_RGB = self.conv2_RGB(c1_RGB)  # N,512,48,48
        G2 = self.attention_feature2(torch.cat((c2_RGB, c2_Flow), dim=1))
        G2 = F.adaptive_avg_pool2d(torch.sigmoid(G2), 1)
        c2_RGB = G2[:, 0, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * c2_RGB
        c2_Flow = G2[:, 1, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * c2_Flow
        temp = c2_Flow.mul(self.atten_flow_channel_2(c2_Flow))
        temp = temp.mul(self.atten_flow_spatial_2(temp))
        c2_RGB = c2_RGB+temp

        c3_RGB = self.conv3_RGB(c2_RGB)  # N,1024,24,24
        G3 = self.attention_feature3(torch.cat((c3_RGB, c3_Flow), dim=1))
        G3 = F.adaptive_avg_pool2d(torch.sigmoid(G3), 1)
        c3_RGB = G3[:, 0, :, :].unsqueeze(1).repeat(1,1024, 1, 1) * c3_RGB
        c3_Flow = G3[:, 1, :, :].unsqueeze(1).repeat(1, 1024, 1, 1) * c3_Flow
        temp = c3_Flow.mul(self.atten_flow_channel_3(c3_Flow))
        temp = temp.mul(self.atten_flow_spatial_3(temp))
        c3_RGB = c3_RGB+temp

        c4_RGB = self.conv4_RGB(c3_RGB)  # N,2048,12,12
        G4 = self.attention_feature4(torch.cat((c4_RGB, c4_Flow), dim=1))
        G4 = F.adaptive_avg_pool2d(torch.sigmoid(G4), 1)
        c4_RGB = G4[:, 0, :, :].unsqueeze(1).repeat(1, 2048, 1, 1) * c4_RGB
        c4_Flow = G4[:, 1, :, :].unsqueeze(1).repeat(1, 2048, 1, 1) * c4_Flow
        temp = c4_Flow.mul(self.atten_flow_channel_4(c4_Flow))
        temp = temp.mul(self.atten_flow_spatial_4(temp))
        c4_RGB = c4_RGB+temp
        ################################PAFEM######################################
        c4_RGB_512 = self.gate_RGB4(c4_RGB)  # 512
        c3_RGB_512 = self.gate_RGB3(c3_RGB)  # 256
        c2_RGB_512 = self.gate_RGB2(c2_RGB)  # 128
        c1_RGB_512 = self.gate_RGB1(c1_RGB)  # 64
        c0_RGB_512 = self.gate_RGB0(c0_RGB)  # 32

        c4_Flow_512 = self.gate_Flow4(c4_Flow)  # 512
        c3_Flow_512 = self.gate_Flow3(c3_Flow)  # 256
        c2_Flow_512 = self.gate_Flow2(c2_Flow)  # 128
        c1_Flow_512 = self.gate_Flow1(c1_Flow)  # 64
        c0_Flow_512 = self.gate_Flow0(c0_Flow)  # 32

        batch, channel, h, w = c4_RGB_512.shape
        M = h * w
        Flow_features4 = c4_Flow_512.view(batch, channel, M).permute(0, 2, 1)
        RGB_features4 = c4_RGB_512.view(batch, channel, M)
        p_4 = torch.matmul(Flow_features4, RGB_features4)
        p_4 = F.softmax((channel ** -.5) * p_4, dim=-1)
        feats_RGB4 = torch.matmul(p_4, RGB_features4.permute(0, 2, 1)).permute(0, 2, 1).view(batch, channel, h, w)

        E4_RGB = self.fuse4_RGB(torch.cat((c4_RGB_512, feats_RGB4), dim=1))  # 512->256 256->1
        E4_Flow = self.fuse4_Flow(torch.cat((c4_Flow_512, feats_RGB4), dim=1))  # 256->128
        channel_4 = self.channel4(E4_Flow)
        c4_attention = self.spatial4(channel_4 * E4_Flow)  # 4,1,12,12
        output1 = self.output1(c4_attention * E4_RGB + feats_RGB4)  # 512->256

        c3 = F.interpolate(output1, size=c3_RGB_512.size()[2:], mode='bilinear',  align_corners=True)  # 256
        E3_Flow = self.fuse3_Flow(torch.cat((c3_Flow_512, c3), dim=1))  # 256->128
        channel_3 = self.channel3(E3_Flow)
        c3_attention = self.spatial3(channel_3 * E3_Flow)  # 4,1,24,24
        E3_RGB = self.fuse3_RGB(torch.cat((c3_RGB_512, c3), dim=1))  # 256->128
        output2 = self.output2(c3_attention * E3_RGB + c3)  # 256->128

        c2 = F.interpolate(output2, size=c2_RGB_512.size()[2:], mode='bilinear',  align_corners=True)  # 256
        E2_Flow = self.fuse2_Flow(torch.cat((c2_Flow_512, c2), dim=1))  # 256->128
        channel_2 = self.channel2(E2_Flow)
        c2_attention = self.spatial2(channel_2 * E2_Flow)  # 4,1,24,24
        E2_RGB = self.fuse2_RGB(torch.cat((c2_RGB_512, c2), dim=1))  # 256->128
        output3 = self.output3(c2_attention * E2_RGB + c2)  # 256->128

        c1 = F.interpolate(output3, size=c1_RGB_512.size()[2:], mode='bilinear',  align_corners=True)  # 256
        E1_Flow = self.fuse1_Flow(torch.cat((c1_Flow_512, c1), dim=1))  # 256->128
        channel_1 = self.channel1(E1_Flow)
        c1_attention = self.spatial1(channel_1 * E1_Flow)  # 4,1,24,24
        E1_RGB = self.fuse1_RGB(torch.cat((c1_RGB_512, c1), dim=1))  # 256->128
        output4 = self.output4(c1_attention * E1_RGB + c1)  # 256->128

        c0 = F.interpolate(output4, size=c0_RGB_512.size()[2:], mode='bilinear',  align_corners=True)  # 256
        E0_Flow = self.fuse0_Flow(torch.cat((c0_Flow_512, c0), dim=1))  # 256->128
        channel_0 = self.channel0(E0_Flow)
        c0_attention = self.spatial0(channel_0 * E0_Flow)  # 4,1,24,24
        E0_RGB = self.fuse0_RGB(torch.cat((c0_RGB_512, c0), dim=1))  # 256->128
        output = self.output5(c0_attention * E0_RGB + c0)  # 256->128

        output = F.interpolate(output, size=input.size()[2:], mode='bilinear',  align_corners=True)
        output = torch.sigmoid(output)

        

        return output, c4_attention, c3_attention, c2_attention, c1_attention, c0_attention


if __name__ == "__main__":
    model = AMCNet()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    depth = torch.autograd.Variable(torch.randn(4, 1, 384, 384))
    flow = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    output,a,b,c,d,e = model(input, flow)
    print(output.shape)
