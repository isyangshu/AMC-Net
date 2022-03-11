import torch

from MATNet import MATNet
from AMCNet import AMCNet
import time
import thop
inputRes = (384, 384)

torch.cuda.set_device(device=0)

AMCNet = AMCNet().cuda()
AMCNet.train(False)
input = torch.randn(1, 3, 384, 384).cuda()
flow = torch.randn(1, 3, 384, 384).cuda()
start_time = time.time()
for i in range(1000):
    with torch.no_grad():
        a, _, _, _, _, _ = AMCNet.forward(input, flow)
end_time = time.time()
print(end_time-start_time)
print('AMC-time:', (end_time-start_time)/1000)
# macs, params = thop.profile(AMCNet, inputs=(input,flow, ))
# print('macs_AMC-Net:', macs)
# print('params_AMC-Net', params)

MATNet = MATNet().cuda()
MATNet.train(False)
input = torch.randn(1, 3, 384, 384).cuda()
flow = torch.randn(1, 3, 384, 384).cuda()
start_time = time.time()
for i in range(1000):
    with torch.no_grad():
        a = MATNet.forward(input, flow)
end_time = time.time()
print(end_time-start_time)
print('MAT-time:', (end_time-start_time)/1000)
# macs, params = thop.profile(MATNet, inputs=(input,flow, ))
# print('macs_MAT-Net:', macs)
# print('params_MAT-Net', params)

