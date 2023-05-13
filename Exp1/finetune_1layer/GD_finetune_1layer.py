import os
import numpy as np
import argparse
import copy
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from modules import Net, Train, Test
from data import *
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=1)   # last k layers to be finetuned
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--model-path', type=str, default='./saved_models/')
    parser.add_argument('--result-path', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=12000)
    parser.add_argument('--plot-epoch', type=int, default=2000)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    # Detect devices
    use_cuda = torch.cuda.is_available()                     # check if GPU exists
    device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    Path(args.result_path).mkdir(parents=True, exist_ok=True)

    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}
    test_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}

    # load target data
    target_data = TensorDataset(t_tilde.to(device), x_tilde.to(device))
    target_loader = DataLoader(target_data, **params)

    # test data
    target_test_loader = DataLoader(target_data, **test_params)

    # set finetune model g_GD
    g = Net(input_size=t.size(1), hidden_size1=64, hidden_size2=64, num_class=1).to(device)
    g.load_state_dict(torch.load(os.path.join(args.model_path, 'pretrain_best.pth')))

    # Select layers other than last k layers to be fixed
    fixed_module = torch.nn.Sequential(*list(g.children())[:-args.k])

    # Fix previous layers
    for param in fixed_module.parameters():
        param.requires_grad = False
        print(param.size(), 'fixed!')
    print(f'Finetune last {args.k} layer(s) [first {len(torch.nn.Sequential(*list(g.children()))) - args.k} pretrained layers fixed!]')

    optimizer = torch.optim.Adam(g.parameters(), lr=args.lr)

    # record finetune process
    eval_finetune_losses = []
    best_weights = copy.deepcopy(g.state_dict())
    best_epoch = 0
    best_eval_loss = 1000000.0

    # training /testing
    with tqdm(total=args.epochs, unit='step') as t:
        for epoch in range(args.epochs):
            # train model
            finetune_loss = Train(g, device, target_loader, optimizer)
            t.set_postfix({'GD Finetune loss': f'{finetune_loss:.4f}'})

            # test model
            eval_finetune_loss, all_X, all_y, all_y_pred = Test(g, device, target_test_loader)
            eval_finetune_losses.append(eval_finetune_loss)
            t.set_postfix({'GD Finetune eval loss': f'{eval_finetune_loss:.4f}'})
            t.update(1)

            if eval_finetune_loss < best_eval_loss:
                best_epoch = epoch
                best_eval_loss = eval_finetune_loss
                best_weights = copy.deepcopy(g.state_dict())

            if (epoch % args.plot_epoch) == 0:
                plt.figure()
                plt.scatter(all_X, all_y, s=3, label='label')
                plt.scatter(all_X, all_y_pred, s=3, label='pred')
                plt.legend(['label', 'pred'], loc="upper left")
                plt.title('Target signal prediction')
                plt.xlabel('t', fontsize=14)
                plt.ylabel('x(t)', fontsize=14)
                plt.tight_layout()
                plt.show()
                plt.close()

    print(f'Best finetune epoch: {best_epoch}, loss: {best_eval_loss:.4f}')
    torch.save(best_weights, os.path.join(args.model_path, f'grad_finetune_{args.k}layer_best.pth'))

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(all_X, all_y, s=3, label='label')
    plt.scatter(all_X, all_y_pred, s=3, label='pred')
    plt.legend()
    plt.title('Target time series')
    plt.xlabel('t', fontsize=14)
    plt.ylabel('x(t)', fontsize=14)
    plt.legend(['label', 'pred'], loc="upper left")

    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, args.epochs + 1), np.array(eval_finetune_losses))
    plt.title("GD finetune loss")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.tight_layout()
    img = os.path.join(args.result_path, f"GD_{args.k}layer_finetune_loss_prediction.png")
    plt.savefig(img, dpi=600)
    plt.show()
