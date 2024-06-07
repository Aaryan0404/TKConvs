import torch
from tqdm import trange
import numpy as np
import sys
from torch import nn

# Simulate a conv as an implicit GEMM

N = 1

# input height, width
H = 256
W = 256

# output height, width
P = 256
Q = 256

assert(P == H)
assert(Q == W)

# channels
C = 64

# num filters
K = 64
# filter height, width
R = 3
S = 3

torch.random.manual_seed(42)
cuda_device = torch.device('cuda')

ref_operation = nn.Sequential(
        nn.Conv2d(C, K, 3, stride=1, padding=1, bias=False, padding_mode="zeros")
    )
ref_operation.to(dtype=torch.float16, device=cuda_device)

weights = ref_operation.state_dict()['0.weight']
x = torch.rand(N, C, H, W, dtype=torch.float16, device=cuda_device)

output_ref = ref_operation(x)

input_data = x.permute(0, 2, 3, 1)
weights_data = weights
output_data = output_ref.permute(0, 2, 3, 1)

print("txt input shape: ", input_data.shape)
print("txt weights shape: ", weights_data.shape)
print("txt output shape: ", output_data.shape)

# Perform the convolution with an implicit GEMM
# M = B * img_H * img_W
# N = C
# K = C * kernel_H * kernel_W

# Perform the implict GEMM

# First, unfold the input to im2col format
unfolded_input = torch.nn.functional.unfold(input_data.permute(0, 3, 1, 2), (R, S), padding=1)
input_gemm = unfolded_input.permute(0, 2, 1).contiguous().view(-1, C * R * S)

# Reshape weights to match the GEMM operation
weights_gemm = weights_data.view(K, -1).permute(1, 0).contiguous()

# Perform the GEMM operation
output_gemm = input_gemm @ weights_gemm

# print shapes
print("A matrix shape: ", input_gemm.shape)
print("B matrix shape: ", weights_gemm.shape)
print("Output shape: ", output_gemm.shape)

# Reshape the output to match the expected output shape
output_gemm = output_gemm.view(N, P, Q, K).permute(0, 3, 1, 2).contiguous()
output_gemm = output_gemm.permute(0, 2, 3, 1)

# print out the difference between the two outputs
print("Verified implicit GEMM wrt PyTorch Conv - max diff: ", torch.max(torch.abs(output_gemm - output_data)).item())

fn = f'randn.txt'
with open(fn, 'w') as f:
    af = input_data.to(torch.float32).flatten().detach().cpu().numpy()
    bf = weights_data.to(torch.float32).flatten().detach().cpu().numpy()
    cf = output_data.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(input_data.shape[0] * input_data.shape[1] * input_data.shape[2] * input_data.shape[3]):
        f.write(repr(af[i]))
        f.write(' ')
    for i in trange(weights_data.shape[0] * weights_data.shape[1] * weights_data.shape[2] * weights_data.shape[3]):
        f.write(repr(bf[i]))
        f.write(' ')
    for i in trange(output_data.shape[0] * output_data.shape[1] * output_data.shape[2] * output_data.shape[3]):
        f.write(repr(cf[i]))
        f.write(' ')

    
print(f"Generated {fn}")

