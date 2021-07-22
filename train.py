#!/usr/bin/env python

import argparse
import gc
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import AdamW, lr_scheduler
from tqdm.auto import tqdm

from dataset import AtmaDataset
from model import freeze_backbone_params
from model import create_model
from util import SimpleLogger
from util import AverageMeter
from util import seed_everything

parser = argparse.ArgumentParser(description='AtmaCup11 Training')
parser.add_argument('--checkpoint', 'checkpoint.pth', help='learning rate')
parser.add_argument('--lr', default=1e-3, help='learning rate')
parser.add_argument('--wd', default=1e-2, help='weight decay')
parser.add_argument('--seed', default=2929, help='random seed')
parser.add_argument('--device', default='cuda', help='tensor device')
parser.add_argument('--batch_size', default=64, help='batch size')
parser.add_argument('--epochs', default=100, help='epochs')
parser.add_argument('--display_epochs', default=10, help='epochs to display graph')
parser.add_argument('--folds', default=5, help='number of cv folds')
parser.add_argument('--arch', default='vit_small', help='vit architecture')
parser.add_argument('--patch_size', default=16, help='vit patch size')


def main():
    args = parser.parse_args()
    
    train_df = pd.read_csv('train.csv')

    if args.seed is not None:
        seed_everything(args.seed)

    fold = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2929)
    cv = list(fold.split(X=train_df, y=train_df['target'], groups=train_df['art_series_id']))[:args.folds]
    train_meta_df = create_metadata(train_df)

    for fold, (idx_tr, idx_valid) in enumerate(cv):
        model = create_model(args)
        model.to(args.device)

        # linear layer 以外を freeze
        freeze_backbone_params(model)

        print(f'Start fold {fold}:')

        run_fold(
            model=model, 
            train_df=train_meta_df.iloc[idx_tr], 
            valid_df=train_meta_df.iloc[idx_valid], 
            y_valid=train_meta_df['target'].values[idx_valid],
            fold=fold,
            args=args,
        )

    torch.save(model, f'model_{fold}.pt')
    del model

def run_fold(
    model: nn.Module,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_valid: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    ) -> np.ndarray:
    """
    train / valid に分割されたデータで学習と同時に検証を行なう
    """

    train_dataset = AtmaDataset(meta_df=train_df)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2
    )

    # 検証用の方は is_train=False にしてデータ拡張オフにする
    valid_dataset = AtmaDataset(meta_df=valid_df, is_train=False)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size * 4, num_workers=2)

    criterion = nn.MSELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd)
    scheduler = lr_scheduler.OneCycleLR(optimizer, epochs=args.epochs, steps_per_epoch=len(train_loader),
                                        max_lr=1.0e-3, pct_start=0.1, anneal_strategy='cos',
                                        div_factor=1.0e+3, final_div_factor=1.0e+3
                                        )
    train_logger = SimpleLogger()
    valid_logger = SimpleLogger()

    best_score = 100
    for epoch in range(1, args.epochs):
        print(f'start {epoch}')

        train_loss = train_epoch(model, criterion, optimizer, scheduler, train_loader, epoch, train_logger, args)
        val_loss = valid_epoch(model, valid_loader, y_valid, criterion, epoch, valid_logger, args)
        valid_logger.add(val_loss)

        content = f"""
                Fold:{fold}, Epoch:{epoch}, {optimizer.param_groups[0]['lr']:.7}\n
                Train Loss(RMSE):{np.sqrt(train_loss):0.4f}\n
                Valid Loss(RMSE):{val_loss:0.4f}\n
        """
        print(content)

        if val_loss < best_score:
            print('Best score achieved. Saving model...')
            best_score = val_loss
            torch.save(model, f'best_model_{fold}.pt')

        if epoch % args.display_epoch == args.display_epoch - 1:
            train_logger.plot('Train Loss', file_name=f'train_log_fold{fold}.png')
            valid_logger.plot('Valid Loss', file_name=f'valid_log_fold{fold}.png')


def train_epoch(
    model, criterion, optimizer, scheduler,
    loader, epoch, logger, args):

    losses = AverageMeter()

    model.train()

    t = tqdm(loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, (image, label) in enumerate(t):
        image = image.to(args.device)
        label = label.to(args.device).float()
        batch_size = image.shape[0]
        optimizer.zero_grad()

        y_pred = model(image)
        loss = criterion(y_pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), batch_size)

        # display
        avg_loss = np.sqrt(losses.avg) # RSME
        t.set_description(f"Train E: {epoch} - Upstream Loss: {losses.avg:0.4f}")

        # log
        logger.add(loss.item())

    t.close()
    gc.collect()
    return loss


def valid_epoch(model, loader, y_valid, criterion, epoch, logger, args):
    model.eval()
    losses = AverageMeter()
    predicts = []
    t = tqdm(loader)

    with torch.no_grad():
        for i, (image, label) in enumerate(t):
            image = image.to(args.device)
            label = label.to(args.device).float()
            batch_size = label.shape[0]

            y_pred = model(image)
            predicts.extend(y_pred.data.cpu().numpy())

            loss  = criterion(y_pred, label)
            logger.add(loss.item())
            t.set_description(f"Valid Epoch {epoch} - Loss(RSME): {np.sqrt(losses.avg):0.4f}")

        pred = np.array(predicts).reshape(-1)
        score = calculate_metrics(y_valid, pred)

    t.close()
    gc.collect()
    return score


def calculate_metrics(y_true, y_pred) -> dict:
    """正解ラベルと予測ラベルから指標を計算する"""    
    return mean_squared_error(y_true, y_pred) ** .5


def create_metadata(input_df):
    def to_img_path(object_id):
        return os.path.join('./photos', f'{object_id}.jpg')

    target = 'target'
    out_df = input_df[['object_id']].copy()
    out_df['object_path'] = input_df['object_id'].map(to_img_path)

    if target in input_df:
        out_df[target] = input_df[target]
    
    return out_df


if __name__ == '__main__':
    main()
