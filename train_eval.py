import time
import os
import math
import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util_functions import PyGGraph_to_nx
from torch_geometric.utils.convert import to_networkx
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import evaluation_fun

def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 必须为0
        pin_memory=False,
        persistent_workers=False
    )

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


#train model
def train_multiple_epochs(train_dataset,
                          test_dataset,
                          model,
                          epochs,
                          batch_size,
                          lr,
                          lr_decay_factor,
                          lr_decay_step_size,
                          weight_decay,
                          ARR=0, 
                          test_freq=1, 
                          logger=None,
                          continue_from=None, 
                          res_dir=None,
                          force_cpu=False):
    print(f"[Training Check] Device: {next(model.parameters()).device}")
    if not force_cpu:
        model.to(device)

    rmses = []

    train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size, shuffle=False)

    model.to(device).reset_parameters()

    #optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    batch_pbar = len(train_dataset) >= 100000

    t_start = time.perf_counter()
    if not batch_pbar:
        pbar = tqdm(range(start_epoch, epochs + start_epoch))
    else:
        pbar = range(start_epoch, epochs + start_epoch)
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, device, regression=True, ARR=ARR,
                           show_progress=batch_pbar, epoch=epoch)
        if epoch % test_freq == 0:
            rmses.append(eval_rmse(model, test_loader,epoch, device, show_progress=batch_pbar))
        else:
            rmses.append(np.nan)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
        }
        if not batch_pbar:
            pbar.set_description(
                'Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values())
            )
        else:
            print('Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values()))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if epoch == 20:
            model_name = os.path.join('./model_save/','model_checkpoint{}.pth'.format(epoch))
            torch.save(model.state_dict(),model_name)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    duration = t_end - t_start

    print('Final Test RMSE: {:.6f}, Duration: {:.6f}'.
          format(rmses[-1],
                 duration))

    return rmses[-1]


def test_once(test_dataset,
              model,
              batch_size,
              logger=None, 
              ensemble=False, 
              checkpoints=None):

    test_loader = get_dataloader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    t_start = time.perf_counter()
    if ensemble and checkpoints:
        rmse = eval_rmse_ensemble(model, checkpoints, test_loader, device, show_progress=True)
    else:
        rmse = eval_rmse(model, test_loader, device, show_progress=True)
    t_end = time.perf_counter()
    duration = t_end - t_start
    print('Test Once RMSE: {:.6f}, Duration: {:.6f}'.format(rmse, duration))
    epoch_info = 'test_once' if not ensemble else 'ensemble'
    eval_info = {
        'epoch': epoch_info,
        'train_loss': 0,
        'test_rmse': rmse,
        }
    if logger is not None:
        logger(eval_info, None, None)
    return rmse


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train(model, optimizer, loader, device, regression=True, ARR=0,
          show_progress=False, epoch=None):
    model.train()
    total_loss = 0
    if show_progress:
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        optimizer.zero_grad()
        data = data.to(device)
        out,x1= model(data)
        
        if regression:
            loss = F.mse_loss(out, data.y.view(-1))
        else:
            loss = F.nll_loss(out, data.y.view(-1))
        if show_progress:
            pbar.set_description('Epoch {}, batch loss: {}'.format(epoch, loss.item()))
        if ARR != 0:
            for gconv in model.convs:
                w = torch.matmul(
                    gconv.att, 
                    gconv.basis.view(gconv.num_bases, -1)
                ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


#
def eval_loss(model, loader, epoch,device, regression=False, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    print("pbar",len(pbar))

    test_results=[]
    latent_feature=[]
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out,x1= model(data)
            test_results.extend(out.cpu().numpy().tolist())
            latent_feature.extend(x1.cpu().numpy().tolist())

        if regression:
            loss += F.mse_loss(out, data.y.view(-1), reduction='sum').item()
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()

    print("len_test:",len(test_results))
    print("len_latent:", len(latent_feature))
    if len(test_results)>140:
        print(np.array(test_results).shape,type(np.array(test_results)))

    #####
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, 'Dataset'))
    pair = np.load(os.path.join(dataset_path, 'LncPair.npy'))
    pair_new = np.load(os.path.join(dataset_path, 'LncPair_new.npy'))
    # pair = np.load(os.path.join(dataset_path, 'MiPair.npy'))
    # pair_new = np.load(os.path.join(dataset_path, 'MiPair_new.npy'))
    test_index = np.where((pair_new == 2) | (pair_new == -30))
    # val_index = np.where((pair_new == -1) | (pair_new == -10))
    adj_test = np.zeros(pair.shape)
    adj_test[test_index] = pair[test_index]
    test_label = np.array(adj_test[test_index])

    results=np.array(test_results).flatten()
    torch.cuda.empty_cache()
    ####
    

    cur_AUC, cur_AUPR, cur_NDCG10, cur_MAP, cur_MRR, cur_MRR10, cur_ROC = \
        evaluation_fun.evaluation_all(results,test_label)
    print("AUC:",cur_AUC,"AUPR:",cur_AUPR, flush=True)

    np.save(os.path.join(dataset_path, 'LncEvaluation.npy'), results)
    # np.save(os.path.join(dataset_path, 'MiEvaluation.npy'), results)
    return loss / len(loader.dataset)


def eval_rmse(model, loader, epoch, device, show_progress=False):
    mse_loss = eval_loss(model, loader, epoch, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse


def eval_loss_ensemble(model, checkpoints, loader, device, regression=False, show_progress=False):
    loss = 0
    Outs = []
    for i, checkpoint in enumerate(checkpoints):
        if show_progress:
            print('Testing begins...')
            pbar = tqdm(loader)
        else:
            pbar = loader
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        outs = []
        if i == 0:
            ys = []
        for data in pbar:
            data = data.to(device)
            if i == 0:
                ys.append(data.y.view(-1))
            with torch.no_grad():
                out,x1= model(data)
                outs.append(out)
        if i == 0:
            ys = torch.cat(ys, 0)
        outs = torch.cat(outs, 0).view(-1, 1)
        Outs.append(outs)
        print("out1",Outs.shape)
    Outs = torch.cat(Outs, 1).mean(1)
    print("out2",Outs.shape)
    
    if regression:
        loss += F.mse_loss(Outs, ys, reduction='sum').item()
    else:
        loss += F.nll_loss(Outs, ys, reduction='sum').item()
    torch.cuda.empty_cache()
    return loss / len(loader.dataset)


def eval_rmse_ensemble(model, checkpoints, loader, device, show_progress=False):
    mse_loss = eval_loss_ensemble(model, checkpoints, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse