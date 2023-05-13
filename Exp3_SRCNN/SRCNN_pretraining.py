import argparse
import os
import copy
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from datasets import image_paths, TrainDataset
from utils import AverageMeter, calc_psnr
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/CUFED_blurred_patches/')
    parser.add_argument('--output-model-dir', type=str, default='./pretrained_models_SR/')
    parser.add_argument('--scale', type=str, default='3')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-name', type=str, default='SRCNN_02')
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # check / create output folders
    Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set deblur model
    exec(f"from models import {args.model_name} as model")  # choose model
    model = model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    # data path
    train_input, train_label = image_paths(scale=args.scale, mode='train', path=args.image_path)
    test_input, test_label = image_paths(scale=args.scale, mode='test', path=args.image_path)

    # data loaders
    train_dataset = TrainDataset(input_path=train_input, label_path=train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_dataset = TrainDataset(input_path=test_input, label_path=test_label)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    N = len(train_loader) * args.batch_size

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for count, (X, y) in enumerate(train_loader):
                X = X.unsqueeze(1).to(device)   # blurred images as inputs
                y = y.unsqueeze(1).to(device)   # de-blurred images as outputs

                x_latent, y_pred = model(X)
                loss = criterion(y_pred, y)
                epoch_losses.update(loss.item(), len(X))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(X))

        model.eval()
        epoch_psnr = AverageMeter()

        for X, y in eval_loader:
            X = X.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)

            with torch.no_grad():
                y_pred = model(X)[1].clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(y_pred, y), len(X))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print(f'best epoch: {best_epoch}, psnr: {best_psnr}')
    torch.save(best_weights, os.path.join(args.output_model_dir, f'{args.model_name}_scale{args.scale}_best.pth'))
