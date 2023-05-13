import os
import numpy as np
import torch


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

def grad(F, x, h):
    ''' F: R^n -> R
    input x size: (samples, feature dim)
    h : small scalar '''
    b,t,f = x.shape
    x = x.view(-1,f)
#     device_=x.device
    F = F.cpu()
    e = torch.eye(x.size(1))  # coordinate basis: e = {e1, ..., e_n}, each e_j \in R^n
    z = torch.zeros_like(e)
    dF = (F((x.unsqueeze(1) + (h * e)))[1] - F(x.unsqueeze(1)+z)[1])
    return dF.view(b,t,f,-1).transpose(2,3)


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


def align(x_old, y_old, x_new, y_new, p=2):
    """ find most similar samples in x_old & y_old
        (index 0 is assumed sample dimension) """
    # x_old, y_old = torch.rand(17, 19, 7), torch.rand(17, 19, 7)  # test examples
    # x_new, y_new = torch.rand(31, 19, 7), torch.rand(31, 19, 7)

    # ||(x old)_i - (x new)_j ||: shape=(N_old, N_new)
    dist_x = torch.cdist(x_old.view(x_old.shape[0], -1), x_new.view(x_new.shape[0], -1), p)
    dist_y = torch.cdist(y_old.view(y_old.shape[0], -1), y_new.view(y_new.shape[0], -1), p)

    # consider input-label product distance
    dist = dist_x + dist_y

    # find index of minimal dist in N_old
    ind_min = torch.argmin(dist, dim=0)

    return x_old[ind_min], y_old[ind_min]


def finetune(x, y, x_tilde, y_tilde, f, method='exact', alignment=False, p_norm=2):
    """f: pretrained model
       F_k: pretrained f but removing last layer"""
    device = x.device

    # align data (find most similar sample pairs)
    x, y, x_tilde, y_tilde, f = x.cpu(), y.cpu(), x_tilde.cpu(), y_tilde.cpu(), f.cpu()
    if alignment or (x.shape[0] != x_tilde.shape[0]):
        x, y = align(x, y, x_tilde, y_tilde, p=p_norm)

    # x & x_tilde are aligned now (same for y & y_tilde)
    z = f(x)[0]              # latent of old data
    z_tilde = f(x_tilde)[0]  # latent of new data
    W = f.last_layer.weight.data.T   # last layer weights
    
    N, T, dim_z = z_tilde.shape   # shape z = (sample, time, latent dim)
    z_tilde = z_tilde.reshape(N * T, dim_z)
    z = z.reshape(N * T, dim_z)

    _, _, dim_y = y_tilde.shape
    y_tilde = y_tilde.reshape(N * T, dim_y)
    y = y.reshape(N * T, dim_y)

    if method == 'exact':
        dz = z_tilde - z
        y_intrinsic = dz @ W
    elif method == 'perturb':
        dx = x_tilde - x    # input difference
        Jf = Jacobi_fully_parallel(f, x, h=0.001)
        ddf = Jacobi_partial_parallel(f, x, h=0.001)
        dddf = Jacobi_v0(f, x, h=0.001)  # df size: (N samples, x feature-dim)
        y_intrinsic = torch.einsum('ijk,ik->ij', Jf, dx.reshape(x.size(0), -1)).unsqueeze(-1)  # Jf(x_i) * dx_i
        df = grad(f, x.unsqueeze(0), h=0.001).squeeze(0)
        y_intrinsic = torch.bmm(df, dx.unsqueeze(-1)).squeeze()

    dy = y_tilde - y   # label deviation
    q = dy - y_intrinsic + y - z @ W
    
    # finding weight adjustment by pseudo_inv(z) * q
    dW = np.linalg.pinv(z_tilde.data.cpu().numpy()) @ q.data.cpu().numpy()
    dW = torch.from_numpy(dW.T).to(device)
    f = f.to(device)

    return dW