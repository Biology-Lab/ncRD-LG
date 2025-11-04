import time
import json
import numpy as np
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from main import args, LocalGL, MyDynamicDataset, import_data, evaluation_fun
import torch
import numpy as np
import scipy.sparse as ssp
import os
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from train_eval import *
from models import *
import evaluation_fun
import pandas as pd
import matplotlib.pyplot as plt
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def setup_device():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.init()
    return device

device = setup_device()

# Parameter grid configuration
param_grid = {
    'hop': [1, 2, 3, 4],
    'epochs': [10, 20, 30, 40],
    'lr': [0.001, 0.005, 0.01],
    'latent_dim': [[128,64,32,1], [64,32,1], [32,1]]
}



def initialize_data_components():
    RNA_feature, Drug_feature, Pair, Pair_new = import_data()
    
    train_index = np.where((Pair_new == 1) | (Pair_new == -20))
    val_index = np.where((Pair_new == -1) | (Pair_new == -10))
    
    adj_train = np.copy(Pair)
    adj_train[val_index] = 0
    train_A_csr = ssp.csr_matrix(adj_train)
    train_label = np.array(adj_train[train_index])
    
    adj_val = np.zeros(Pair.shape)
    adj_val[val_index] = Pair[val_index]
    val_label = np.array(adj_val[val_index])
    
    RNA_features = np.array(RNA_feature)
    Drug_features = np.array(Drug_feature)
    class_values = [0, 1]
    
    return {
        'train_A_csr': train_A_csr,
        'train_index': train_index,
        'train_label': train_label,
        'val_index': val_index,
        'val_label': val_label,
        'RNA_features': RNA_features,
        'Drug_features': Drug_features,
        'class_values': class_values,
        'Pair': Pair,
        'Pair_new': Pair_new
    }

# 在main()中初始化
data_components = initialize_data_components()
globals().update(data_components)  # 将变量设为全局可用


def extract_metrics():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, 'Dataset'))
    results = np.load(os.path.join(dataset_path, 'LncEvaluation.npy'))
    # results = np.load(os.path.join(dataset_path, 'MiEvaluation.npy'))
    LncPair = np.load(os.path.join(dataset_path, 'LncPair.npy'))
    # MiPair = np.load(os.path.join(dataset_path, 'MiPair.npy'))
    LncPair_new = np.load(os.path.join(dataset_path, 'LncPair_new.npy'))
    # MiPair_new = np.load(os.path.join(dataset_path, 'MiPair_new.npy'))
    Lncval_index = np.where((LncPair_new == -1) | (LncPair_new == -10))
    Lncval_label = np.array(LncPair[Lncval_index])
    
    return evaluation_fun.evaluation_all(results.flatten(), Lncval_label)

def run_single_experiment(params, base_args):

    args = deepcopy(base_args)
    data_combo = (args.data_name, args.data_appendix, 'valmode')
    
    args.hop = params['hop']
    args.epochs = params['epochs']
    args.lr = params['lr']
    args.lr_decay_step_size = max(1, params['epochs']//4)
    
    train_graphs = MyDynamicDataset(
        'data/{}{}/{}/train'.format(*data_combo),
        train_A_csr,
        train_index,
        train_label,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        RNA_features,
        Drug_features,
        class_values
    )

    val_graphs = MyDynamicDataset(
        'data/{}{}/{}/val'.format(*data_combo),
        train_A_csr,
        val_index,
        val_label,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        RNA_features,
        Drug_features,
        class_values
    )
    
    model = LocalGL(
        train_graphs,
        latent_dim=params['latent_dim'],
        num_relations=len(class_values),
        num_bases=0,
        regression=True,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        side_features=args.use_features,
        n_side_features=RNA_features.shape[1] + Drug_features.shape[1],
        multiply_by=1
    ).to(device)

    # train
    start_time = time.time()

    try:
        start_time = time.time()
        train_multiple_epochs(
            train_graphs,
            val_graphs,
            model,
            args.epochs,
            args.batch_size,
            args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            ARR=args.ARR,
            test_freq=args.epochs,
            force_cpu=False
        )
        
        # 评估
        metrics = extract_metrics()
        duration = time.time() - start_time
        
        print(f"\n=== Results for {params} ===")
        print(f"AUC: {metrics[0]:.4f}, AUPR: {metrics[1]:.4f}")
        
        return {
            'params': params,
            'AUC': metrics[0],
            'AUPR': metrics[1],
            'duration': duration
        }
    finally:
        torch.cuda.empty_cache()

all_results = []

if __name__ == "__main__":
    data_components = initialize_data_components()
    globals().update(data_components)
    
    for params in ParameterGrid(param_grid):
        try:
            all_results.append(run_single_experiment(params, args))
        except Exception as e:
            print(f"Failed with params {params}: {str(e)}")
            continue
    
    # save
    if all_results:
        df = pd.DataFrame(all_results)
        df['Composite_Score'] = (df['AUC'] + df['AUPR']) / 2
        df.to_csv('grid_search_results.csv', index=False)
    
        best = max(all_results, key=lambda x: (x['AUC'] + x['AUPR'])/2)
        print(f"\n Best params: {best['params']}")
        print(f"AUC: {best['AUC']:.4f}, AUPR: {best['AUPR']:.4f}")
    else:
        print("No valid results obtained!")