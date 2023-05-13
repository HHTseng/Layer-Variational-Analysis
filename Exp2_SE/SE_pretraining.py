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
from datasets import AdaptationData
from utils import AverageMeter, get_filepaths, write_score
from pathlib import Path
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/mnt/Datasets/transfer/data/')
    parser.add_argument('--score-path', type=str, default='./results/')
    parser.add_argument('--output-model-dir', type=str, default='./pretrained_models_SE/')
    parser.add_argument('--enhanced-path', type=str, default='./enhanced/')
    parser.add_argument('--data', type=str, default='source')   # options: source/target/clean
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model-name', type=str, default='BLSTM')  # DDAE/LSTM/BLSTM
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument("--evaluate-only", default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # check / create output folders
    Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.score_path).mkdir(parents=True, exist_ok=True)
    Path(args.enhanced_path).mkdir(parents=True, exist_ok=True)

    # train, test data path
    Train_path = {'source': os.path.join(args.data_path, 'train_log1p_pt/source/'),
                  'target': os.path.join(args.data_path, 'train_log1p_pt/target/'),
                  'clean': os.path.join(args.data_path, 'train_log1p_pt/clean/')}

    Test_path = {'source': os.path.join(args.data_path, 'test/source/'),
                 'target': os.path.join(args.data_path, 'test/target/'),
                 'clean': os.path.join(args.data_path, 'test/clean/')}

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set denoise model
    exec(f"from models import {args.model_name} as model")  # choose model
    f = model().to(device)     # pretrain model f
    criterion = nn.MSELoss()
    optimizer = optim.Adam(f.parameters(), lr=args.lr, weight_decay=0)

    # training SE models
    if not args.evaluate_only:
        # data loaders
        train_paths, eval_paths = train_test_split(get_filepaths(Train_path[args.data], '.pt'), test_size=0.1, random_state=999)

        N_train = int(len(train_paths) // 1000)   # samples for training
        N_eval = int(len(eval_paths) // 100)      # samples for evaluation
        train_dataset = AdaptationData(train_paths, Train_path['clean'], N=N_train)
        eval_dataset = AdaptationData(eval_paths, Train_path['clean'], N=N_eval)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        best_weights = copy.deepcopy(f.state_dict())
        best_epoch = 0
        best_eval_loss = 1000000.0

        for epoch in range(args.num_epochs):
            f.train()
            epoch_train_loss = AverageMeter()
            with tqdm(total=len(train_loader), desc=f'Train epoch: {epoch}/{args.num_epochs - 1}', unit='step') as t:
                for count, (X, y) in enumerate(train_loader):
                    X = X.to(device)   # noisy speech (input)
                    y = y.to(device)   # clean speech (label)

                    x_latent, y_pred = f(X)
                    loss = criterion(y_pred, y)
                    epoch_train_loss.update(loss.item(), len(X))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    t.set_postfix({'loss': f'{epoch_train_loss.avg:.4f}'})
                    t.update(1)

            f.eval()
            epoch_eval_loss = AverageMeter()
            with tqdm(total=len(eval_loader), desc=f'[evaluation]', unit='step') as q:
                for X, y in eval_loader:
                    X = X.to(device)
                    y = y.to(device)

                    with torch.no_grad():
                        y_pred = f(X)[1]
                        epoch_eval_loss.update(criterion(y_pred, y).item(), len(X))

                    q.set_postfix({'loss': f'{epoch_eval_loss.avg:.4f}'})
                    q.update(1)

            if epoch_eval_loss.avg < best_eval_loss:
                best_epoch = epoch
                best_eval_loss = epoch_eval_loss.avg
                best_weights = copy.deepcopy(f.state_dict())

        print(f'best epoch: {best_epoch}, loss: {best_eval_loss}')
        torch.save(best_weights, os.path.join(args.output_model_dir, f'{args.model_name}_best.pth'))

    # reload best pretrained model f
    f.load_state_dict(torch.load(os.path.join(args.output_model_dir, f'{args.model_name}_best.pth')))
    print(f'{args.model_name}_best.pth reloaded!')

    # Evaluate on (source, target) domain
    f.eval()
    for domain in ['source', 'target']:
        score_file = os.path.join(args.score_path, f'{args.model_name}_{domain}.csv')
        write_score(f, device, criterion, Test_path[domain], Test_path['clean'], domain, score_file)
