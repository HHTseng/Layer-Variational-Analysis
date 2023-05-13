import os, pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from utils import AverageMeter, calc_psnr

ch_in, ch_out = 32, 1
ker, stride = 5, 1
height, width = 33, 33

def _unfold(X):
    unfold = nn.Unfold(ker, padding=(ker - 1) // 2, stride=stride)
    X_unfold = unfold(X).transpose(1, 2) 
    return X_unfold

def _fold(X_unfold):
    batch = X_unfold.shape[0]
    X = X_unfold.transpose(1, 2).reshape(batch, ch_out, ceil(height / stride), ceil(width / stride))
    return X
    
def flatten(conv_model):
    W = conv_model.weight.reshape(ch_out, -1).transpose(0, 1)
    return W

def unflatten(W):
    Wn = W.transpose(0, 1).reshape(ch_out, ch_in, ker, ker)
    return Wn

def cat_to_1hot(y, num_class):
    y_onehot = torch.zeros(y.size(0), num_class, device=y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot

def Jacobi_fully_parallel(F, x, h):
    ''' F: R^n -> R^m
    input x size: (samples, feature dim)
    h : small scalar '''
    
    original_device = x.device
    F = F.cpu()
    x = x.cpu()
    N, c, w, l = x.size()
    m = w * l
#     m = list(F.children())[-1].out_features  # output dimension of F
    x = x.reshape(N, -1)   # flatten all features of x
    e = torch.eye(x.size(1), dtype=bool, device=x.device)  # coordinate basis: e = {e1, ..., e_n}, each e_j \in R^n

    ''' Compute JF in fully parallel fashion (requires lots of memory, hence on CPU)
        JF := [ F(x_j + h * e_j) - F(x_j) ] / h,  (j = 1,..., n) 
        JF size: (N, m: output dim of F, dim of input features flattened)
    '''
    JF = F((x.unsqueeze(1) + h * e).reshape(N * c * w * l, c, w, l))[1] - F((x.unsqueeze(1) + 0 * e).reshape(N * c * w * l, c, w, l))[1]
    JF = (JF / h).reshape(N, c * w * l, m).transpose_(1, 2).to(original_device)

    # put models back to original device
    F = F.to(original_device)
    x = x.to(original_device)
    return JF

def Jacobi_partial_parallel(F, x, h):
    ''' Compute JF in semi-parallel fashion requiring less memory
        JF := [ F(x_j + h * e_j) - F(x_j) ] / h,  (j = 1,..., n)
        JF size: (N, m: output dim of F, dim of input features flattened)
        F: R^n -> R^m
        input x size: (samples, feature dim)
        h : small scalar '''
    original_device = x.device
    F = F.cpu()
    x = x.cpu()
    N, c, w, l = x.size()
    m = list(F.children())[-1].out_features  # output dimension of F
    x = x.reshape(N, -1)  # flatten input features
    e = torch.eye(x.size(1), dtype=bool, device=x.device)  # coordinate basis: e = {e1, ..., e_n}, each e_j \in R^n
    JF = torch.zeros((N, c * w * l, m), device=x.device)

    # compute dF(x_i) \in R^n for each sample i
    for i in range(N):
        JF[i] = (F((x[i].unsqueeze(0) + h * e).reshape(c * w * l, c, w, l))[1] - F((x[i].unsqueeze(0) + 0 * e).reshape(c * w * l, c, w, l))[1]) / h

    # put models back to original device
    F = F.to(original_device)
    x = x.to(original_device)
    return JF.transpose_(1, 2).to(original_device)

def Jacobi_v0(F, x, h):
    ''' Compute JF in semi-parallel fashion requiring less memory
        JF := [ F(x_j + h * e_j) - F(x_j) ] / h,  (j = 1,..., n)
        JF size: (N, m: output dim of F, dim of input features flattened)
        F: R^n -> R^m
        input x size: (samples, feature dim)
        h : small scalar '''
    original_device = x.device
    F = F.cpu()
    x = x.cpu()
    N, c, w, l = x.size()
    m = list(F.children())[-1].out_features  # output dimension of F
    x = x.reshape(N, -1)  # flatten input features
    e = torch.eye(x.size(1), dtype=bool, device=x.device)  # coordinate basis: e = {e1, ..., e_n}, each e_j \in R^n
    JF = torch.zeros((N, c * w * l, m), device=x.device)

    # compute dF(x_i) \in R^n for each sample i
    for i in range(N):
        D = torch.zeros((x.size(1), m), device=x.device)
        for j in range(x.size(1)):
            D[j, :] = (F((x[i] + h * e[j]).reshape(c, w, l).unsqueeze(0)) - F(x[i].reshape(c, w, l).unsqueeze(0))) / h
        JF[i] = D

    # put models back to original device
    F = F.to(original_device)
    x = x.to(original_device)
    return JF.transpose_(1, 2).to(original_device)


def LVA_finetune(x, y, dx, dy, f, method='exact'):
    """f: original model
       F_k: original model f but removing last layer"""
    
    z = f(x)[0]             # latent of old data
    z_tilde = f(x + dx)[0]  # latent of new data

    W = flatten(f.conv3)   # flatten conv kernels to fully-connected weights
    z_unfold = _unfold(z)
    z_tilde_unfold = _unfold(z_tilde)
    y_unfold = y.reshape(x.shape[0], -1, 1)
    
    if method == 'exact':
        dz = z_tilde_unfold - z_unfold
        y_intrinsic = dz @ W
    elif method == 'perturb':
        Jf = Jacobi_fully_parallel(f, x, h=0.001)
        # Jf = Jacobi_partial_parallel(f, x, h=0.001)   # different computational method for Jacobian f
        # Jf = Jacobi_v0(f, x, h=0.001)  # df size: (N samples, x feature-dim)  # different computational method for Jacobian f
        y_intrinsic = torch.einsum('ijk,ik->ij', Jf, dx.reshape(x.size(0), -1)).unsqueeze(-1)  # Jf(x_i) * dx_i

    q = dy - y_intrinsic + y_unfold - z_unfold @ W

    # finding weight adjustment by pseudo_inv(z) * q
    z_tilde_unfold = z_tilde_unfold.reshape(1, -1, z_tilde_unfold.shape[-1])
    q = q.reshape(1, -1, 1)
    dW_np = np.linalg.pinv(z_tilde_unfold.data.cpu().numpy()) @ q.data.cpu().numpy()
    dW = torch.from_numpy(dW_np).to(x.device)
    return dW
