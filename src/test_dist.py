import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

t = np.random.randn(50,128,128,64).astype(np.float32)

x = torch.from_numpy(t).permute(0,3,1,2)

model1 = nn.Sequential(
    nn.Conv2d(64, 128, 3, stride=1, padding=1, dilation=1),
    nn.Conv2d(128, 256, 3, stride=1, padding=1, dilation=1),
)

model2 = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(64, 16, kernel_size=(1, 1)),
        nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
        nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1)),
        nn.Conv2d(16, 128, kernel_size=(1, 1)),
    ),
    nn.Sequential(
        nn.Conv2d(128, 32, kernel_size=(1, 1)),
        nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
        nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
        nn.Conv2d(32, 256, kernel_size=(1, 1)),
    ),
)

torch.set_num_threads()

iters = 2

t1 = time.time()
for i in range(iters):
    y=model1(x)
t2 = time.time()
m1t = t2-t1

t1 = time.time()
for i in range(iters):
    y=model2(x)
t2 = time.time()
m2t = t2-t1

print(m1t, m2t)