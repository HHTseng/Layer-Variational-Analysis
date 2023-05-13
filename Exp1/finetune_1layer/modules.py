import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_class, bias1=False, bias2=True, bias3=False):
        super().__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(input_size, hidden_size1, bias=bias1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=bias2)
        self.fc3 = nn.Linear(hidden_size2, num_class, bias=bias3)

    def forward(self, x):
        z1 = Func.relu(self.fc1(x))
        z2 = Func.relu(self.fc2(z1))
        z3 = self.fc3(z2)
        return z1, z2, z3


def Train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        z1, z2, y_pred = model(X)   # output has dim = (batch, number of classes)
        loss = Func.mse_loss(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def Test(model, device, test_loader):
    model.eval()
    all_X, all_y, all_y_pred = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            z1, z2, y_pred = model(X)
            loss = Func.mse_loss(y_pred, y)

            all_X.extend(X)
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    all_X = torch.stack(all_X, dim=0)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    total_loss = Func.mse_loss(all_y_pred, all_y)

    return total_loss.item(), all_X.cpu(), all_y.cpu(), all_y_pred.cpu()


def grad(F, x, h):
    ''' F: R^n -> R
    input x size: (samples, feature dim)
    h : small scalar '''
    e = torch.eye(x.size(1), device=x.device)  # coordinate basis: e = {e1, ..., e_n}, each e_j \in R^n
    dF = torch.zeros_like(x, device=x.device)

    # compute dF(x_i) \in R^n for each sample i
    for i in range(x.size(0)):
        dF[i] = ( F((x[i] + h * e).to(x.device))[-1] - F(x[i].to(x.device))[-1] ) / h
    return dF


def LVA_finetune(x, y, dx, dy, f, method='exact'):
    """f: original model
       F_k: original model f but removing last layer"""

    z = f(x)[1]             # latent of old data
    z_tilde = f(x + dx)[1]  # latent of new data
    W = f.fc3.weight.T      # last layer weights

    if method == 'exact':
        dz = z_tilde - z
        y_intrinsic = dz @ W

    elif method == 'perturb':
        Jf = grad(f, x, h=0.001)   # df size: (N samples, x feature-dim)
        y_intrinsic = torch.einsum('ij,ij->i', Jf, dx.reshape(x.size(0), -1)).unsqueeze(-1)  # Jf(x_i) * dx_i

    # transferal residue
    q = dy - y_intrinsic + y - z @ W

    # compute weight adjustment by pseudo_inv(z) * q
    dW = np.linalg.pinv(z_tilde.data.cpu().numpy()) @ q.data.cpu().numpy()
    dW = torch.from_numpy(dW.astype(np.float32)).to(x.device)

    return dW