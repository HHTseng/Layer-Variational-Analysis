import math
import torch

torch.manual_seed(937)

# Time series (signal): (t, x(t))
t = torch.arange(-1, 1, 0.001).reshape(-1, 1) + 0.0005
x = torch.sin(5 * math.pi * t).reshape(-1, 1)   # t -> x(t)

N = t.size(0)

# Target domain signal
dt = 0.05 * torch.normal(1.5, 0.8, size=(N, 1))
dx = 0.03 * (2 * torch.rand(size=(N, 1)) - 1)

# scale
r = torch.arange(1, 1.8, 0.0004).reshape(-1, 1)

t_tilde = t + dt
x_tilde = r * x + dx
