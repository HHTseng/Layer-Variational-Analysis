import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Func
from modules import Net, LVA_finetune
from data import *
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=1)  # last k layers to be finetuned
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--model-path', type=str, default='./saved_models/')
    parser.add_argument('--LVA-method', type=str, default='exact')  # 'exact' or 'perturb'
    parser.add_argument('--result-path', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--plot-epoch', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
    Path(args.result_path).mkdir(parents=True, exist_ok=True)

    t, x = t.to(device), x.to(device)
    t_tilde, x_tilde = t_tilde.to(device), x_tilde.to(device)
    dt, dx = t_tilde - t, x_tilde - x

    # reload models
    f = Net(input_size=t.size(1), hidden_size1=64, hidden_size2=64, num_class=1).to(device)
    f.load_state_dict(torch.load(os.path.join(args.model_path, 'pretrain_best.pth')))
    f.eval()

    # LVA finetune model: g_LVA
    g_LVA = Net(input_size=t.size(1), hidden_size1=64, hidden_size2=64, num_class=1).to(device)
    g_LVA.load_state_dict(torch.load(os.path.join(args.model_path, f'pretrain_best.pth')))
    g_LVA.eval()

    # grad finetune model: g_GD
    g_GD = Net(input_size=t.size(1), hidden_size1=64, hidden_size2=64, num_class=1).to(device)
    g_GD.load_state_dict(torch.load(os.path.join(args.model_path, f'grad_finetune_{args.k}layer_best.pth')))
    g_GD.eval()

    # compare weights & bias
    pretrained_module = nn.Sequential(*list(f.children())[:-args.k])
    GD_fixed_module = nn.Sequential(*list(g_GD.children())[:-args.k])

    # Fix previous layers
    for layer, (f_params, g_params) in enumerate(zip(pretrained_module.parameters(), GD_fixed_module.parameters())):
        if torch.is_nonzero((f_params - g_params).sum()):
            print(f'Layer {layer + 1} is not fixed!')
    print(f'Gradient Descent (GD) finetune last {args.k} layer(s) [first {len(nn.Sequential(*list(g_GD.children()))) - args.k} pretrained layers fixed!]')


    # Finetune weights by LVA method
    W = g_LVA.fc3.weight.detach().T
    dW = LVA_finetune(t, x, dt, dx, g_LVA,  method=args.LVA_method)  # LVA finetuning weight

    # compare LVA & GD finetune models
    z_tilde = f(t_tilde)[-(args.k + 1)]  # last k^th layer latent
    x_LVA_pred = z_tilde @ (W + dW)
    x_GD_pred = z_tilde @ g_GD.fc3.weight.T

    loss_LVA = Func.mse_loss(x_LVA_pred, x_tilde, reduction='sum')
    loss_GD = Func.mse_loss(x_GD_pred, x_tilde, reduction='sum')

    print(f'[{args.k}-layer finetune] GD loss: {loss_GD.item():.4f} | LVA loss: {loss_LVA.item():.4f} ')

    # plot
    t = t.cpu().numpy()
    x = x.cpu().numpy()
    t_tilde = t_tilde.cpu().numpy()
    x_tilde = x_tilde.cpu().numpy()
    x_LVA_pred = x_LVA_pred.detach().cpu().numpy()
    x_GD_pred = x_GD_pred.detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 4))
    plt.subplot(131)
    plt.scatter(t_tilde, x_tilde, s=3, label='target')
    plt.scatter(t, x, s=3, c='red', alpha=0.2, label='source')
    plt.legend()
    plt.title('(source, target) signals')
    plt.xlabel('t', fontsize=14)
    plt.ylabel('x(t)', fontsize=14)
    plt.legend(['target', 'source'], loc="upper left")

    plt.subplot(132)
    plt.scatter(t_tilde, x_tilde, s=3, label='label')
    plt.scatter(t_tilde, x_GD_pred, s=3, label='GD pred')
    plt.legend()
    plt.title(f'GD finetune {args.k}-layer target signal')
    plt.xlabel('t', fontsize=14)
    plt.ylabel('x(t)', fontsize=14)
    plt.legend(['label', 'GD predict'], loc="upper left")

    plt.subplot(133)
    plt.scatter(t_tilde, x_tilde, s=3, label='label')
    plt.scatter(t_tilde, x_LVA_pred, s=3, label='LVA pred')
    plt.legend()
    plt.title(f'LVA finetune {args.k}-layer target signal')
    plt.xlabel('t', fontsize=14)
    plt.ylabel('x(t)', fontsize=14)
    plt.legend(['label', 'LVA predict'], loc="upper left")
    plt.tight_layout()
    img = os.path.join(args.result_path, f"{args.k}layer_finetune_prediction.png")
    plt.savefig(img, dpi=600)
    plt.show()
