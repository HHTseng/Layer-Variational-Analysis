import os, argparse, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from modules import *
from datasets import image_paths, TrainDataset
from pathlib import Path
from utils import calc_psnr, test_SET14


score_path = './Result/'   # for writing scores on txt files
Path(score_path).mkdir(parents=True, exist_ok=True)

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0" if use_cuda else "cpu") # use CPU or GPU
k = 1    # last k layers to be finetuned

def get_loss(loader, model):
    criterion = torch.nn.MSELoss()
    losses = 0
    psnrs = 0
    count = 0
    model.eval()
    for count, (X, y) in enumerate(loader):
        X = X.unsqueeze(1).to(device)
        y = y.unsqueeze(1).to(device)
        count += X.shape[0]
        x_latent, y_pred = model(X)
        loss = criterion(y_pred, y).cpu()
        psnr = calc_psnr(y_pred, y).cpu()
        losses += loss.item()
        psnrs += psnr.item()
    return losses / count, psnr / count


def compute_scores(g, method, image_path, finetune_loader, eval_loader=None):
    train_loss, train_psnr = get_loss(finetune_loader, g)

    if eval_loader:
        test_loss, test_psnr = get_loss(eval_loader, g)           # test on same dataset as training
    else:
        test_loss, test_psnr = test_SET14(g, args.finetuned_scale, path=image_path)  # test on different dataset (eg. SET14 images)

    # write scores
    print(f'{method} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
    scorefile.write(f'{method} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
    scorefile.write(f'{method} | train psnr: {train_psnr:.4f} | test psnr: {test_psnr:.4f}')


parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str, default='/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/CUFED_blurred_patches/')
parser.add_argument('--test-image-path', type=str, default='/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/Set14')
parser.add_argument('--pretrained-model-path', type=str, default='./pretrained_models_SR/')
parser.add_argument('--finetuned-model-path', type=str, default='./finetuned_models_SR/')
parser.add_argument('--model-name', type=str, default='SRCNN_02')
parser.add_argument('--LVA-method', type=str, default='exact')    # 'exact' or 'perturb'
parser.add_argument('--N-samples', type=int, default=100)
parser.add_argument('--pretrained-scale', type=str, default='3')
parser.add_argument('--finetuned-scale', type=str, default='6')
parser.add_argument('--batch-size', type=int, default=1)
args = parser.parse_args()

print(f'Transfer learning samples: {args.N_samples}')

# data path
train_input1, train_label1 = image_paths(scale=args.pretrained_scale, mode='train', path=args.image_path)  # pretrain images
train_input2, train_label2 = image_paths(scale=args.finetuned_scale, mode='train', path=args.image_path)   # finetune images
test_input, test_label = image_paths(scale=args.finetuned_scale, mode='test', path=args.image_path)        # finetune images

# pytorch data loaders
train_dataset1 = TrainDataset(input_path=train_input1, label_path=train_label1, N=args.N_samples)  # pretraine data
train_dataset2 = TrainDataset(input_path=train_input2, label_path=train_label2, N=args.N_samples)  # finetune data
eval_dataset = TrainDataset(input_path=test_input, label_path=test_label)                          # finetune test data

use_cuda = torch.cuda.is_available()
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 0, 'pin_memory': False, 'drop_last': True} if use_cuda else {}
test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'pin_memory': False, 'drop_last': True} if use_cuda else {}
finetune_loader = DataLoader(dataset=train_dataset2, **params)    # finetune data
eval_loader = DataLoader(dataset=eval_dataset, **test_params)  # finetune test data


# load data
X, y = train_dataset1.input.unsqueeze(1).to(device), train_dataset1.label.unsqueeze(1).to(device)  # old data
X_tilde = train_dataset2.input.unsqueeze(1).to(device)   # new data

# data difference
dx = X_tilde - X
dy = 0    # due to same SR label

score_file = os.path.join(score_path, f'{args.model_name}_scale{args.pretrained_scale}to{args.finetuned_scale}_{args.N_samples}samples.txt')
scorefile = open(score_file, "w")


# reload pretrained model
exec(f"from models import {args.model_name} as model")

# reload Gradient Descent finetuned model
g_grad = model().to(device)   # transfer learning by gradient descent
g_grad.load_state_dict(torch.load(os.path.join(args.finetuned_model_path, f'{args.model_name}_scale{args.pretrained_scale}to{args.finetuned_scale}_{args.N_samples}samples_grad.pth'), map_location=torch.device('cpu')))

# reload pretrain model for LVA
g_LVA = model().to(device)
g_LVA.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, f'{args.model_name}_scale{args.pretrained_scale}_best.pth')))

# LVA finetune weights
W = flatten(g_LVA.conv3)                                          # pretrained weight
dW = LVA_finetune(X, y, dx, dy, g_LVA, method=args.LVA_method)    # LVA finetuning weight
W_analytic = W + dW                                               # new weight (viewed as fully-connected)
g_LVA.conv3.weight = torch.nn.Parameter(unflatten(W_analytic))    # new weight (viewed as conv-kernels)

torch.save(copy.deepcopy(g_LVA.state_dict()),
           os.path.join(args.finetuned_model_path, f'{args.model_name}_scale{args.pretrained_scale}to{args.finetuned_scale}_{args.N_samples}samples_LVA.pth'))

# compare scores
compute_scores(g_LVA, method=f'LVA ({args.LVA_method})', image_path=args.test_image_path, finetune_loader=finetune_loader, eval_loader=None)
compute_scores(g_grad, method=f'  GD method', image_path=args.test_image_path, finetune_loader=finetune_loader, eval_loader=None)

scorefile.flush()
scorefile.close()
