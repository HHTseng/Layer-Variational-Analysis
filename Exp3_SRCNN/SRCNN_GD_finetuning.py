import os, argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import copy
from tqdm import tqdm
from datasets import image_paths, TrainDataset
from utils import AverageMeter, calc_psnr


parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str, default='/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/CUFED_blurred_patches/')
parser.add_argument('--pretrained-model-path', type=str, default='./pretrained_models_SR/')
parser.add_argument('--finetuned-model-path', type=str, default='./finetuned_models_SR/')
parser.add_argument('--model_name', type=str, default='SRCNN_02')
parser.add_argument('--N-samples', type=int, default=100)
parser.add_argument('--pretrained-scale', type=str, default='3')
parser.add_argument('--finetuned-scale', type=str, default='6')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=1000)
args = parser.parse_args()

# check / create output folders
Path(args.finetuned_model_path).mkdir(parents=True, exist_ok=True)

SEED = 14823
torch.manual_seed(SEED)
cudnn.deterministic = True
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': False, 'drop_last': True} if use_cuda else {}
test_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': False, 'drop_last': True} if use_cuda else {}

# finetuning model params
k = 1               # last k layers to be finetuned
learning_rate = 1e-4

# data path
train_input, train_label = image_paths(scale=args.finetuned_scale, mode='train', path=args.image_path)
test_input, test_label = image_paths(scale=args.finetuned_scale, mode='test', path=args.image_path)

# data loaders
finetune_data = TrainDataset(input_path=train_input, label_path=train_label, N=args.N_samples)
eval_data = TrainDataset(input_path=test_input, label_path=test_label)

finetune_loader = DataLoader(dataset=finetune_data, **params)
eval_loader = DataLoader(dataset=eval_data, **test_params)


# set model
exec(f"from models import {args.model_name} as model")
model = model().to(device)
model.load_state_dict(torch.load(f'{args.pretrained_model_path}{args.model_name}_scale{args.pretrained_scale}_best.pth'))  # load pretrained model
criterion = torch.nn.MSELoss()

# Select layers other than last k layers to be fixed
fixed_module = torch.nn.Sequential(*list(model.children())[:-k])

# Fix previous layers
for param in fixed_module.parameters():
    param.requires_grad = False
    print(param.size(), 'fixed!')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_loss = 100
# start training
for epoch in range(args.num_epochs):
    # train, test model
    model.train()
    epoch_losses = AverageMeter()
    with tqdm(total=(len(finetune_data) - len(finetune_data) % args.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
        for count, (X, y) in enumerate(finetune_loader):
            X = X.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)

            x_latent, y_pred = model(X)
            loss = criterion(y_pred, y)
            epoch_losses.update(loss.item(), len(X))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(X))
        
    if epoch_losses.avg < best_loss:
        best_epoch = epoch
        best_loss = epoch_losses.avg
        best_weights = copy.deepcopy(model.state_dict())

    
model.eval()
epoch_psnr = AverageMeter()
eval_loss = AverageMeter()
for X, y in eval_loader:
    X = X.unsqueeze(1).to(device)
    y = y.unsqueeze(1).to(device)

    with torch.no_grad():
        y_pred = model(X)[1].clamp(0.0, 1.0)
        loss = criterion(y_pred, y)

    eval_loss.update(loss.item(), len(X))
    epoch_psnr.update(calc_psnr(y_pred, y), len(X))

print(f'eval psnr: {epoch_psnr.avg:.2f} / eval loss: {eval_loss.avg:.6f}')
save_path = os.path.join(args.finetuned_model_path, f'{args.model_name}_scale{args.pretrained_scale}to{args.finetuned_scale}_{args.N_samples}samples_grad.pth')
torch.save(best_weights, save_path)

print('finetune done!')
