import os
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import copy
from tqdm import tqdm
from datasets import AdaptationData
from utils import get_filepaths, write_score
from sklearn.model_selection import train_test_split
from modules import *


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='/mnt/Datasets/transfer/data/')
parser.add_argument('--pretrained-model-path', type=str, default='./pretrained_models_SE/')
parser.add_argument('--finetuned-model-path', type=str, default='./finetuned_models_SE/')
parser.add_argument('--score-path', type=str, default='./results/')
parser.add_argument('--model-name', type=str, default='DDAE')   # DDAE/LSTM/BLSTM
parser.add_argument('--ntype', type=str, default='babycry')  # babycry, BELL, SIREN
parser.add_argument('--SNR', type=str, default='-1')   # target SNR: -1, 1
parser.add_argument('--N-source', type=int, default=30000)
parser.add_argument('--N-target', type=int, default=100)
parser.add_argument('--source-batch-size', type=int, default=500)
parser.add_argument('--target-batch-size', type=int, default=20)
parser.add_argument('--num-epochs', type=int, default=3)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()


# check / create output folders
Path(args.pretrained_model_path).mkdir(parents=True, exist_ok=True)
Path(args.finetuned_model_path).mkdir(parents=True, exist_ok=True)
Path(args.score_path).mkdir(parents=True, exist_ok=True)


# train, test data path
Train_path = {'source': os.path.join(args.data_path, 'train_log1p_pt/source/'),
              'target': os.path.join(args.data_path, f'train_log1p_pt/target/{args.ntype}/{args.SNR}/'),
              'clean': os.path.join(args.data_path, 'train_log1p_pt/clean/')}

Test_path = {'source': os.path.join(args.data_path, 'test/source/'),
             'target': os.path.join(args.data_path, f'test/target/{args.ntype}/{args.SNR}/'),
             'clean': os.path.join(args.data_path, 'test/clean/')}

SEED = 14823
torch.manual_seed(SEED)
cudnn.deterministic = True
use_cuda = torch.cuda.is_available()                     # check if GPU exists
device = torch.device("cuda:1" if use_cuda else "cpu")   # use CPU or GPU
source_params = {'batch_size': args.source_batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True, 'drop_last': True} if use_cuda else {}
target_params = {'batch_size': args.target_batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True, 'drop_last': True} if use_cuda else {}
test_params = {'batch_size': args.target_batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True, 'drop_last': True} if use_cuda else {}

# finetuning g_grad params
k = 1               # last k layers to be finetuned
learning_rate = 1e-3

print(f'{args.model_name} adapts from {args.N_source} source samples --> {args.N_target} target samples')
source_train_paths, source_eval_paths = train_test_split(get_filepaths(Train_path['source'], '.pt')[:args.N_source], test_size=0.1, random_state=999)
target_train_paths, target_eval_paths = train_test_split(get_filepaths(Train_path['source'], '.pt')[:args.N_target], test_size=0.1, random_state=157)

source_domain = AdaptationData(source_train_paths, Train_path['clean'])
target_domain = AdaptationData(target_train_paths, Train_path['clean'], target_path=Train_path['target'])
target_test = AdaptationData(target_eval_paths, Train_path['clean'], target_path=Train_path['target'])

source_loader = DataLoader(dataset=source_domain, **source_params)    # pretrained data
target_loader = DataLoader(dataset=target_domain, **target_params)    # finetune data
target_test_loader = DataLoader(dataset=target_test, **test_params)   # finetune test data


# set gradient descent f: g_grad
exec(f"from models import {args.model_name} as model")
g_grad = model().to(device)
g_grad.load_state_dict(torch.load(f'{args.pretrained_model_path}{args.model_name}_best.pth'))  # load pretrained g_grad
criterion = nn.MSELoss()

# Select layers other than last k layers to be fixed
fixed_module = nn.Sequential(*list(g_grad.children())[:-k])

# Fix previous layers
for param in fixed_module.parameters():
    param.requires_grad = False
    print(param.size(), 'fixed!')

grad_optimizer = torch.optim.Adam(g_grad.parameters(), lr=learning_rate)
best_g_grad_loss = 100000
best_g_LVA_loss = 100000
best_g_LVA_OT_loss = 100000

# set LVA model from pretrained f
g_LVA = model().to(device)
g_LVA.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, f'{args.model_name}_best.pth')))

# set LVA-OT model from pretrained f
g_LVA_OT = model().to(device)
g_LVA_OT.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, f'{args.model_name}_best.pth')))

# start training
for epoch in range(args.num_epochs):
    # train, test g_grad
    g_grad.train()
    epoch_g_grad_losses = AverageMeter()
    epoch_g_LVA_losses = AverageMeter()
    epoch_g_LVA_OT_losses = AverageMeter()

    source_iterator = iter(source_loader)
    with tqdm(total=(len(target_domain) - len(target_domain) % args.target_batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
        for X_s2, y_tilde, X_tilde in target_loader:
            try:
                X_s1, y_s1 = next(source_iterator)
            except StopIteration:
                source_iterator = iter(source_loader)
                X_s1, y_s1 = next(source_iterator)

            # source domain samples
            X_s1 = X_s1.to(device)
            X_s2 = X_s2.to(device)
            y_s1 = y_s1.to(device)

            # target domain samples
            X_tilde = X_tilde.to(device)
            y_tilde = y_tilde.to(device)

            # ----------- Finetune weights by LVA method ----------- #
            # dW = finetune(X_s2, y_tilde, X_tilde, y_tilde, g_LVA, method='perturb')  # finetuning weight (by perturbation)

            # finetuning by LVA
            W = g_LVA.last_layer.weight.data       # pretrained weight
            dW = finetune(X_s2, y_tilde, X_tilde, y_tilde, g_LVA, method='exact', alignment=False)
            g_LVA.last_layer.weight = torch.nn.Parameter(W + dW)  # new weight of last layer

            # LVA loss
            _, y_tilde_LVA_pred = g_LVA(X_tilde)
            LVA_loss = criterion(y_tilde_LVA_pred, y_tilde)
            epoch_g_LVA_losses.update(LVA_loss.item(), len(X_tilde))

            # finetuning by LVA WITH OT alignment
            W_OT = g_LVA_OT.last_layer.weight.data
            dW_OT = finetune(X_s1, y_s1, X_tilde, y_tilde, g_LVA_OT, method='exact', alignment=True, p_norm=2)
            g_LVA_OT.last_layer.weight = torch.nn.Parameter(W_OT + dW_OT)  # new weight of last layer

            # LVA-OT loss
            _, y_tilde_LVA_OT_pred = g_LVA_OT(X_tilde)
            LVA_OT_loss = criterion(y_tilde_LVA_OT_pred, y_tilde)
            epoch_g_LVA_OT_losses.update(LVA_OT_loss.item(), len(X_tilde))
            # ----------- Finetune weights by LVA method ----------- #


            # ============  finetune: g_grad  ============ #
            x_tilde_latent, y_tilde_pred = g_grad(X_tilde)
            grad_loss = criterion(y_tilde_pred, y_tilde)
            epoch_g_grad_losses.update(grad_loss.item(), len(X_tilde))

            grad_optimizer.zero_grad()
            grad_loss.backward()
            grad_optimizer.step()
            # ============  finetune: g_grad  ============ #

            t.set_postfix(L_LVA_OT=f"{epoch_g_LVA_OT_losses.avg:.5f}", L_LVA=f"{epoch_g_LVA_losses.avg:.5f}", L_grad=f"{epoch_g_grad_losses.avg:.5f}")
            t.update(len(X_tilde))

    if epoch_g_LVA_losses.avg < best_g_LVA_loss:
        best_LVA_epoch = epoch
        best_g_LVA_loss = epoch_g_LVA_losses.avg
        best_g_LVA_weights = copy.deepcopy(g_LVA.state_dict())

    if epoch_g_LVA_OT_losses.avg < best_g_LVA_OT_loss:
        best_LVA_OT_epoch = epoch
        best_g_LVA_OT_loss = epoch_g_LVA_OT_losses.avg
        best_g_LVA_OT_weights = copy.deepcopy(g_LVA_OT.state_dict())

    if epoch_g_grad_losses.avg < best_g_grad_loss:
        best_grad_epoch = epoch
        best_g_grad_loss = epoch_g_grad_losses.avg
        best_g_grad_weights = copy.deepcopy(g_grad.state_dict())


# save best models
save_LVA_path = os.path.join(args.finetuned_model_path, f'{args.model_name}_target{args.N_target}_source{args.N_source}_{args.ntype}_{args.SNR}_LVA.pth')
save_LVA_OT_path = os.path.join(args.finetuned_model_path, f'{args.model_name}_target{args.N_target}_source{args.N_source}_{args.ntype}_{args.SNR}_LVA_OT.pth')
save_grad_path = os.path.join(args.finetuned_model_path, f'{args.model_name}_target{args.N_target}_source{args.N_source}_{args.ntype}_{args.SNR}_grad.pth')

torch.save(best_g_LVA_weights, save_LVA_path)
torch.save(best_g_LVA_OT_weights, save_LVA_OT_path)
torch.save(best_g_grad_weights, save_grad_path)

# Computing scores
for name in ['LVA', 'LVA_OT', 'grad']:
    # reload finetuned nets
    g = model().to(device)  # transfer learning by gradient descent
    g.load_state_dict(torch.load(os.path.join(args.finetuned_model_path, f'{args.model_name}_target{args.N_target}_source{args.N_source}_{args.ntype}_{args.SNR}_{name}.pth')))
    print(f'{args.model_name}_target{args.N_target}_source{args.N_source}_{args.ntype}_{args.SNR}_{name}.pth reloaded!')
    g.eval()
    for domain in ['source', 'target']:
        score_file = os.path.join(args.score_path, f'{args.model_name}_N{args.N_target}_batch{args.target_batch_size}_epochs{args.num_epochs}_{args.ntype}_{args.SNR}_{name}_{domain}.csv')
        write_score(g, device, criterion, Test_path[domain], Test_path['clean'], domain, score_file)
